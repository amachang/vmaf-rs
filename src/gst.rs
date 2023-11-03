use std::{fmt, error, io::{self, BufReader}, fs::File, path::Path, sync::{Arc, Mutex, Weak, Once, Condvar}, thread, num::TryFromIntError};
use crate::{PixelFormat, Picture};
use gstreamer as gst;
use gstreamer_video as gst_video;
use gstreamer_app as gst_app;
use gst::prelude::*;
use lazy_static::lazy_static;


pub fn supported() -> bool { true }


#[derive(Debug)]
pub enum Error {
    UnsupportedVideoFormat(gst_video::VideoFormat),
    GlibError(String),
    StateChangeError(String),
    FailedConversionInteger(String),
    IoError(io::Error),
}

impl From<glib::BoolError> for Error {
    fn from(err: glib::BoolError) -> Self {
        Error::GlibError(err.message.into_owned())
    }
}

impl From<gst::StateChangeError> for Error {
    fn from(err: gst::StateChangeError) -> Self {
        Error::StateChangeError(format!("Gstreamer element failed to change state: {}", err))
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Self {
        Error::IoError(err)
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl error::Error for Error { }


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
            format => return Err(Error::UnsupportedVideoFormat(format)),
        };
        let n_components = from_c_uint(frame.n_components())?;
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
pub struct PictureStream<R: io::Read + io::Seek + Send + Sync + 'static> {
    _gst: Arc<Gst>,
    data: Option<PictureStreamData<R>>,
}

#[derive(Debug)]
struct PictureStreamData<R: io::Read + io::Seek + Send + Sync + 'static> {
    pipeline: gst::Pipeline,
    appsink: gst_app::AppSink,
    _read: Arc<Mutex<R>>,
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> crate::PictureStream for PictureStream<R> {
    fn next_pic(&mut self) -> Result<Option<Picture>, crate::Error> {
        let Some(data) = self.data.as_ref() else {
            return Ok(None);
        };

        if data.appsink.is_eos() {
            self.destroy_pipeline_if_needed();
            return Ok(None);
        }

        let sample: gst::Sample = data.appsink.pull_sample().map_err(Error::from)?;
        let caps = sample.caps().unwrap();
        let buffer = sample.buffer().unwrap();
        let info = gst_video::VideoInfo::from_caps(caps).map_err(Error::from)?;
        let video_frame = gst_video::VideoFrameRef::from_buffer_ref_readable(&buffer, &info).map_err(Error::from)?;
        let frame = Frame::new(&video_frame)?;

        Ok(Some(Picture::new(&frame)?))
    }
}

impl PictureStream<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>, allow_hwaccel: bool) -> Result<Self, Error> {
        let read = BufReader::new(File::open(path)?);
        Self::new(read, allow_hwaccel)
    }
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> PictureStream<R> {
    pub fn new(mut read: R, allow_hwaccel: bool) -> Result<Self, Error> {
        let gst = {
            let mut lock = GST_REF_COUNTER.lock().unwrap();
            lock.refer()
        };

        let pipeline = gst::Pipeline::default();

        // TODO impl From<io::Error>
        let stream_size = {
            let pos = read.stream_position()?;
            let size = read.seek(io::SeekFrom::End(0))?;
            read.seek(io::SeekFrom::Start(pos))?;
            size
        };

        // gstreamer accept i64
        let stream_size: i64 = match stream_size.try_into() {
            Err(err) => return Err(Error::FailedConversionInteger(format!("Invalid Stream Size: {}", err))),
            Ok(stream_size) => stream_size,
        };

        let appsrc = gst_app::AppSrc::builder()
            .format(gst::Format::Bytes)
            .size(stream_size) // if no stream_size, then some gstreamer element couldn't use the appsrc as seekable
            .stream_type(gst_app::AppStreamType::Seekable)
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

        Self::setup_docedbin_connection(&pipeline, &decodebin, &appsink);

        let read = Arc::new(Mutex::new(read));
        Self::setup_appsrc_stream_connection(appsrc, read.clone());

        pipeline.set_state(gst::State::Playing)?;

        Ok(Self {
            _gst: gst,
            data: Some(PictureStreamData {
                pipeline, appsink, _read: read,
            }),
        })
    }

    fn setup_appsrc_stream_connection(appsrc: gst_app::AppSrc, read: Arc<Mutex<R>>) {
        let running_cond_pair = Arc::new((Mutex::new(false), Condvar::new()));
        let running_cond_pair_for_need_data = Arc::clone(&running_cond_pair);
        let running_cond_pair_for_enough_data = Arc::clone(&running_cond_pair);

        let read_weak = Arc::downgrade(&read);
        let appsrc_weak = appsrc.downgrade();
        thread::spawn(move || {
            // TODO error handling
            let Some(read) = read_weak.upgrade() else {
                return;
            };
            let Some(appsrc) = appsrc_weak.upgrade() else {
                return;
            };
            loop {
                let (running_cond, cvar) = &*running_cond_pair;
                {
                    let mut running_cond = running_cond.lock().unwrap();
                    while !*running_cond {
                        running_cond = cvar.wait(running_cond).unwrap();
                    }
                }

                let size = 4096;
                let mut buffer = gst::Buffer::with_size(size).unwrap();

                let buffer_ref = buffer.get_mut().unwrap();
                let size = {
                    let mut buffer_map = buffer_ref.map_writable().unwrap();
                    let buffer_slice = buffer_map.as_mut_slice();
                    let mut read = read.lock().unwrap();
                    read.read(buffer_slice).unwrap()
                };
                if size == 0 {
                    appsrc.end_of_stream().unwrap();
                    break;
                } else {
                    buffer_ref.set_size(size);
                    appsrc.push_buffer(buffer).unwrap();
                };
            };
        });

        let read_weak_for_seek_data = Arc::downgrade(&read);
        let callbacks = gst_app::AppSrcCallbacks::builder()
            .need_data(move |_appsrc, _size| {
                let (running_cond, cvar) = &*running_cond_pair_for_need_data;
                let mut running_cond = running_cond.lock().unwrap();
                *running_cond = true;
                cvar.notify_all();
            })
            .enough_data(move |_appsrc| {
                let (running_cond, cvar) = &*running_cond_pair_for_enough_data;
                let mut running_cond = running_cond.lock().unwrap();
                *running_cond = false;
                cvar.notify_all();
            })
            .seek_data(move |_appsrc, offset| {
                let Some(read) = read_weak_for_seek_data.upgrade() else {
                    return false
                };
                let mut read = match read.lock() {
                    Err(_) => return false,
                    Ok(read) => read,
                };
                let pos = match read.seek(io::SeekFrom::Start(offset)) {
                    Err(_) => return false,
                    Ok(pos) => pos,
                };
                pos == offset
            })
            .build();
        appsrc.set_callbacks(callbacks);
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

    fn destroy_pipeline_if_needed(&mut self) {
        let Some(data) = self.data.take() else {
            return;
        };

        // ignore error
        let _ = data.pipeline.set_state(gst::State::Null);
    }
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> Drop for PictureStream<R> {
    fn drop(&mut self) {
        self.destroy_pipeline_if_needed();
    }
}

fn from_c_uint<T: TryInto<usize, Error = TryFromIntError>>(u: T) -> Result<usize, Error> {
    match u.try_into() {
        Ok(u) => Ok(u),
        Err(err) => Err(Error::FailedConversionInteger(format!("{:?}", err))),
    }
}

