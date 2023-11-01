use std::{io::{self, BufReader}, fs::File, path::Path, sync::{mpsc, Arc, Mutex, Weak, Once}, thread};
use crate::{Error, PixelFormat, Picture, from_c_uint};
use gstreamer as gst;
use gstreamer_video as gst_video;
use gstreamer_app as gst_app;
use gst::prelude::*;
use lazy_static::lazy_static;

pub fn supported() -> bool { true }

#[derive(Debug)]
struct Gst;

impl Gst {
    fn new() -> Self {
        gst::init().unwrap();
        Self
    }
}

impl Drop for Gst {
    fn drop(&mut self) {
        unsafe { gst::deinit() };
    }
}

#[derive(Debug)]
struct GstRefCounter {
    gst: Weak<Gst>,
}

impl GstRefCounter {
    fn new() -> Self {
        Self {
            gst: Weak::new(),
        }
    }

    fn refer(&mut self) -> Arc<Gst> {
        if let Some(gst) = self.gst.upgrade() {
            gst
        } else {
            let gst = Arc::new(Gst::new());
            self.gst = Arc::downgrade(&gst);
            gst
        }
    }
}

lazy_static! {
    static ref GST_REF_COUNTER: Mutex<GstRefCounter> = Mutex::new(GstRefCounter::new());
}

#[derive(Debug)]
struct Component<'data> {
    data: &'data [u8],
    width: usize,
    height: usize,
    stride: usize,
}

impl<'data> crate::Component for Component<'data> {
    fn data(&self) -> &[u8] {
        self.data
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn stride(&self) -> usize {
        self.stride
    }
}

#[derive(Debug)]
struct Frame<'data> {
    width: usize,
    height: usize,
    pixel_format: PixelFormat,
    depth: usize,
    y_component: Component<'data>,
    u_component: Option<Component<'data>>,
    v_component: Option<Component<'data>>,
}

impl<'data> crate::Frame<Component<'data>> for Frame<'data> {
    fn pixel_format(&self) -> PixelFormat {
        self.pixel_format
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn y_component(&self) -> &Component<'data> {
        &self.y_component
    }
    
    fn u_component(&self) -> Option<&Component<'data>> {
        self.u_component.as_ref()
    }

    fn v_component(&self) -> Option<&Component<'data>> {
        self.v_component.as_ref()
    }
}

impl<'data> Frame<'data> {
    fn new(frame: &'data gst_video::VideoFrameRef<&'data gst::BufferRef>) -> Result<Self, Error> {
        let (pixel_format, depth, (width_subsampling, height_subsampling)): (PixelFormat, usize, (usize, usize)) = match frame.format() {
            gst_video::VideoFormat::Gray8 => (PixelFormat::Yuv400p, 8, (0, 0)),
            gst_video::VideoFormat::Gray16Le => (PixelFormat::Yuv400p, 16, (0, 0)),
            gst_video::VideoFormat::I420 => (PixelFormat::Yuv420p, 8, (2, 2)),
            gst_video::VideoFormat::I42010le => (PixelFormat::Yuv420p, 10, (2, 2)),
            gst_video::VideoFormat::I42012le => (PixelFormat::Yuv420p, 12, (2, 2)),
            gst_video::VideoFormat::Y42b => (PixelFormat::Yuv422p, 8, (2, 1)),
            gst_video::VideoFormat::I42210le => (PixelFormat::Yuv422p, 10, (2, 1)),
            gst_video::VideoFormat::I42212le => (PixelFormat::Yuv422p, 12, (2, 1)),
            gst_video::VideoFormat::Y444 => (PixelFormat::Yuv444p, 8, (1, 1)),
            gst_video::VideoFormat::Y44410le => (PixelFormat::Yuv444p, 10, (1, 1)),
            _ => return Err(Error::InvalidArgument)
        };
        let n_components: usize = frame.n_components().try_into().map_err(|_| Error::InvalidArgument)?;
        assert_eq!(n_components, if pixel_format == PixelFormat::Yuv400p { 1 } else { 3 });

        let pixel_stride = depth.div_ceil(8);
        assert_eq!(pixel_stride, from_c_uint(frame.comp_pstride(0))?);

        let width = from_c_uint(frame.width())?;
        let height = from_c_uint(frame.height())?;

        let y_data = frame.comp_data(0)?;
        let y_stride = from_c_uint(frame.comp_stride(0))?;
        let y_width = from_c_uint(frame.comp_width(0))?;
        let y_height = from_c_uint(frame.comp_height(0))?;
        assert_eq!(depth, from_c_uint(frame.comp_depth(0))?);
        assert_eq!(y_data.len(), y_stride * y_height * pixel_stride);

        let y_component = Component { data: y_data, width: y_width, height: y_height, stride: y_stride };

        let (u_component, v_component) = if pixel_format == PixelFormat::Yuv400p {
            (None, None)
        } else {
            let chroma_width = width.div_ceil(width_subsampling);
            let chroma_height = height.div_ceil(height_subsampling);

            assert_eq!(chroma_width, from_c_uint(frame.comp_width(1))?);
            assert_eq!(chroma_width, from_c_uint(frame.comp_width(2))?);
            assert_eq!(chroma_height, from_c_uint(frame.comp_height(1))?);
            assert_eq!(chroma_height, from_c_uint(frame.comp_height(2))?);

            let u_data = frame.comp_data(1)?;
            let u_stride = from_c_uint(frame.comp_stride(1))?;
            assert_eq!(u_data.len(), u_stride * chroma_height * pixel_stride);

            let v_data = frame.comp_data(2)?;
            let v_stride = from_c_uint(frame.comp_stride(2))?;
            assert_eq!(v_data.len(), v_stride * chroma_height * pixel_stride);

            (
                Some(Component { data: u_data, width: chroma_width, height: chroma_height, stride: u_stride }),
                Some(Component { data: v_data, width: chroma_width, height: chroma_height, stride: v_stride }),
            )
        };

        Ok(Self { width, height, pixel_format, depth, y_component, u_component, v_component })
    }
}

#[derive(Debug)]
pub struct PictureStream {
    _gst: Arc<Gst>,
    data: Option<PictureStreamData>,
}

#[derive(Debug)]
struct PictureStreamData {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    read_thread_handle: thread::JoinHandle<()>,
    read_thread_term_sender: mpsc::Sender<()>,
}

impl crate::PictureStream for PictureStream {
    fn next_pic(&mut self) -> Option<Result<Picture, Error>> {
        let Some(data) = self.data.as_ref() else {
            return None;
        };

        if data.appsink.is_eos() {
            self.destroy_pipeline_if_needed();
            return None;
        }
        let sample = match data.appsink.pull_sample() {
            Err(err) => return Some(Err(Error::from(err))),
            Ok(sample) => sample,
        };
        let caps = sample.caps().unwrap();
        let buffer = sample.buffer().unwrap();
        let info = match gst_video::VideoInfo::from_caps(caps) {
            Err(err) => return Some(Err(Error::from(err))),
            Ok(info) => info,
        };
        let video_frame = match gst_video::VideoFrameRef::from_buffer_ref_readable(&buffer, &info) {
            Err(err) => return Some(Err(Error::from(err))),
            Ok(video_frame) => video_frame,
        };
        let frame = match Frame::new(&video_frame) {
            Err(err) => return Some(Err(err)),
            Ok(frame) => frame,
        };
        Some(Picture::new(&frame))
    }
}

impl PictureStream {
    pub fn from_path(path: impl AsRef<Path>, allow_hwaccel: bool) -> Result<Self, Error> {
        let read = BufReader::new(File::open(path).map_err(Error::IoError)?);
        Self::new(read, allow_hwaccel)
    }

    fn destroy_pipeline_if_needed(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };

        // ignore error
        let _ = data.pipeline.set_state(gst::State::Null);
        let _ = data.read_thread_term_sender.send(());
        let _ = data.read_thread_handle.join();
    }
}

impl PictureStream {
    pub fn new<R: io::Read + Send + 'static>(read: R, allow_hwaccel: bool) -> Result<Self, Error> {
        let gst = {
            let mut lock = GST_REF_COUNTER.lock().unwrap();
            lock.refer()
        };

        let pipeline = gst::Pipeline::default();

        let appsrc = gst_app::AppSrc::builder()
            .caps(&gst::Caps::new_any())
            .format(gst::Format::Bytes)
            .block(true)
            .build();

        let decodebin = gst::ElementFactory::make("decodebin")
            .property("force-sw-decoders", !allow_hwaccel)
            .build()?;

        let appsink = gst_app::AppSink::builder()
            .caps(
                &gst_video::VideoCapsBuilder::new()
                    .format_list(vec![
                        gst_video::VideoFormat::Y44410le,
                        gst_video::VideoFormat::Y444,
                        gst_video::VideoFormat::I42212le,
                        gst_video::VideoFormat::I42210le,
                        gst_video::VideoFormat::Y42b,
                        gst_video::VideoFormat::I42012le,
                        gst_video::VideoFormat::I42010le,
                        gst_video::VideoFormat::I420,
                        gst_video::VideoFormat::Gray16Le,
                        gst_video::VideoFormat::Gray8,
                    ])
                    .build()
            )
            .build();

        pipeline.add_many([&appsrc.upcast_ref(), &decodebin, &appsink.upcast_ref()])?;
        gst::Element::link_many([&appsrc.upcast_ref(), &decodebin])?;

        let (read_thread_term_sender, read_thread_term_receiver) = mpsc::channel();
        let read_thread_handle = Self::setup_appsrc_stream_connection(&appsrc, read, read_thread_term_receiver);
        Self::setup_docedbin_connection(&pipeline, &decodebin, &appsink);

        pipeline.set_state(gst::State::Playing)?;

        Ok(Self {
            _gst: gst,
            data: Some(PictureStreamData {
                pipeline, appsink,
                read_thread_handle, read_thread_term_sender,
            }),
        })
    }

    fn setup_appsrc_stream_connection<R: io::Read + Send + 'static>(appsrc: &gst_app::AppSrc, mut read: R, term_receiver: mpsc::Receiver<()>) -> thread::JoinHandle<()> {
        let buffer_size = 4096;
        let appsrc_weak = appsrc.downgrade();
        thread::spawn(move || {
            // TODO error handling
            let appsrc = appsrc_weak.upgrade().unwrap();
            loop {
                let mut buffer = gst::Buffer::with_size(buffer_size).unwrap();
                let buffer_ref = buffer.get_mut().unwrap();
                let read_size = {
                    let mut buffer_map = buffer_ref.map_writable().unwrap();
                    let buffer_slice = buffer_map.as_mut_slice();
                    read.read(buffer_slice).unwrap()
                };
                if read_size == 0 {
                    appsrc.end_of_stream().unwrap();
                    break;
                };
                buffer_ref.set_size(read_size);
                appsrc.push_buffer(buffer).unwrap();
                match term_receiver.try_recv() {
                    Err(mpsc::TryRecvError::Empty) => continue,
                    Err(mpsc::TryRecvError::Disconnected) => break,
                    Ok(()) => break,
                }
            }
        })
    }

    fn setup_docedbin_connection(pipeline: &gst::Pipeline, decodebin: &gst::Element, appsink: &gst_app::AppSink) {
        let pipeline_weak = pipeline.downgrade();
        let appsink_weak = appsink.downgrade();
        let connected_once = Once::new();
        decodebin.connect_pad_added(move |_, src_pad| {
            connected_once.call_once(|| {
                // See: https://gitlab.freedesktop.org/gstreamer/gstreamer-rs/-/blob/main/examples/src/bin/decodebin.rs
                let (Some(pipeline), Some(appsink)) = (pipeline_weak.upgrade(), appsink_weak.upgrade()) else {
                    return;
                };

                let Some(caps) = src_pad.current_caps() else {
                    return;
                };

                let is_video = {
                    let Some(structure) = caps.structure(0) else {
                        return;
                    };
                    structure.name().starts_with("video/")
                };

                if !is_video {
                    return;
                }

                let appsink_caps = appsink.caps().expect("must have a caps");
                if appsink_caps.can_intersect(caps.as_ref()) {
                    src_pad.link(&appsink.static_pad("sink").expect("must have a static pad")).expect("must be able to connect");
                } else {
                    let videoconvert = gst::ElementFactory::make("videoconvert").build().expect("Gstreamer videoconvert element must be installed");
                    pipeline.add(&videoconvert).expect("videoconvert must be abled to be added to pipeline");
                    gst::Element::link_many([&videoconvert, &appsink.upcast_ref()]).expect("videoconvert must be connected with any raw video caps");
                    videoconvert.sync_state_with_parent().expect("videoconvert must be able to be set state");
                    src_pad.link(&videoconvert.static_pad("sink").expect("must have a static pad")).expect("must be able to connect");
                }
            })
        });
    }
}

impl Drop for PictureStream {
    fn drop(&mut self) {
        self.destroy_pipeline_if_needed();
    }
}

impl From<glib::BoolError> for Error {
    fn from(err: glib::BoolError) -> Self {
        Error::GstreamerError(err.message.into_owned())
    }
}

impl From<gst::StateChangeError> for Error {
    fn from(err: gst::StateChangeError) -> Self {
        Error::GstreamerError(format!("Gstreamer element failed to change state: {}", err))
    }
}

