use std::{fmt, error, io::{self, BufReader}, fs::File, path::Path, sync::{Arc, Mutex, Once, Condvar}, thread, num::TryFromIntError};
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
    FailedToGetBusFromPipeline,
    PictureStreamAlreadyDropped,
    ReceivedErrorMessageFromSinkThread(String),
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

lazy_static! {
    static ref INIT_ONCE: Once = Once::new();
}

pub fn init() {
    INIT_ONCE.call_once(|| {
        let _ = gst::init();
    });
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
pub struct PictureStreamOpts {
    pub allow_hwaccel: bool,
    pub start: Option<gst::ClockTime>,
    pub duration: Option<gst::ClockTime>,
    pub fps: Option<usize>,
}

impl Default for PictureStreamOpts {
    fn default() -> Self {
        Self {
            allow_hwaccel: false,
            start: None,
            duration: None,
            fps: None,
        }
    }
}

#[derive(Debug)]
pub struct PictureStream<R: io::Read + io::Seek + Send + Sync + 'static> {
    data: Option<PictureStreamData<R>>,
}

#[derive(Debug)]
struct PictureStreamData<R: io::Read + io::Seek + Send + Sync + 'static> {
    pipeline: gst::Pipeline,
    _bus: gst::Bus,
    reader_thread_data: Arc<(Mutex<PictureStreamReaderThreadData<R>>, Condvar)>,
    sink_thread_data: Arc<(Mutex<PictureStreamSinkThreadData>, Condvar)>,
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> Drop for PictureStreamData<R> {
    fn drop(&mut self) {
        {
            let (sink_thread_data, cvar) = &*self.sink_thread_data;
            let mut sink_thread_data = sink_thread_data.lock().expect("deny poisoned lock");
            sink_thread_data.dead = true;
            cvar.notify_all();
        }

        {
            let (reader_thread_data, cvar) = &*self.reader_thread_data;
            let mut reader_thread_data = reader_thread_data.lock().expect("deny poisoned lock");
            reader_thread_data.dead = true;
            cvar.notify_all();
        }

        // ignore error
        let _ = self.pipeline.set_state(gst::State::Null);
    }
}

#[derive(Debug)]
struct PictureStreamReaderThreadData<R: io::Read + io::Seek + Send + Sync + 'static> {
    read: R,
    needs_data: bool,
    no_more_data: bool,
    dead: bool,
}

#[derive(Debug, Clone)]
struct PictureStreamSinkThreadData {
    per_n_frames: Option<usize>,
    n_pulled_frames: usize,
    error_message: Option<String>,
    new_sample: Option<gst::Sample>,
    eos: bool,
    dead: bool,
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> crate::PictureStream for &mut PictureStream<R> {
    fn next_pic(&mut self) -> Result<Option<Picture>, crate::Error> {
        (*self).next_pic()
    }
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> crate::PictureStream for PictureStream<R> {
    fn next_pic(&mut self) -> Result<Option<Picture>, crate::Error> {
        let Some(data) = self.data.as_ref() else {
            return Ok(None);
        };

        let (sink_thread_data, cvar) = &*data.sink_thread_data;
        let (eos, sample, error_message) = {
            let mut sink_thread_data = sink_thread_data.lock().unwrap();
            while !sink_thread_data.dead && !sink_thread_data.eos && sink_thread_data.error_message.is_none() && sink_thread_data.new_sample.is_none() {
                sink_thread_data = cvar.wait(sink_thread_data).expect("deny poisoned lock");
            };
            let r = (sink_thread_data.eos, sink_thread_data.new_sample.take(), sink_thread_data.error_message.take());
            cvar.notify_all();
            r
        };

        if let Some(error_message) = error_message {
            return Err(crate::Error::from(Error::ReceivedErrorMessageFromSinkThread(error_message)));
        };

        if eos {
            return Ok(None);
        }

        let Some(sample) = sample else {
            return Err(crate::Error::from(Error::PictureStreamAlreadyDropped));
        };

        let caps = sample.caps().unwrap();
        let buffer = sample.buffer().unwrap();
        let info = gst_video::VideoInfo::from_caps(caps).map_err(Error::from)?;
        let video_frame = gst_video::VideoFrameRef::from_buffer_ref_readable(&buffer, &info).map_err(Error::from)?;
        let frame = Frame::new(&video_frame)?;

        Ok(Some(Picture::new(&frame)?))
    }
}

impl PictureStream<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>, opts: PictureStreamOpts) -> Result<Self, Error> {
        let read = BufReader::new(File::open(path)?);
        Self::new(read, opts)
    }
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> PictureStream<R> {
    pub fn new(mut read: R, opts: PictureStreamOpts) -> Result<Self, Error> {
        init();
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
            .property("force-sw-decoders", !opts.allow_hwaccel)
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

            .sync(false)
            .build();

        pipeline.add_many([&appsrc.upcast_ref(), &decodebin, &appsink.upcast_ref()])?;
        gst::Element::link_many([&appsrc.upcast_ref(), &decodebin])?;

        Self::setup_docedbin_connection(&pipeline, &decodebin, &appsink);

        let reader_thread_data = Arc::new(
            (Mutex::new(PictureStreamReaderThreadData { read, needs_data: false, no_more_data: false, dead: false }), Condvar::new())
        );
        Self::setup_reader_thread(appsrc, reader_thread_data.clone());

        let bus = pipeline.bus().unwrap();

        let segment = match (opts.start, opts.duration) {
            (None, None) => None,
            (pos, dur) => {
                let mut segment = gst::Segment::new();
                segment.set_format(gst::Format::Time);
                segment.set_rate(1.0);
                match (pos, dur) {
                    (Some(pos), Some(dur)) => {
                        segment.set_start(pos);
                        segment.set_stop(pos + dur);
                    },
                    (Some(pos), None) => {
                        segment.set_start(pos);
                    },
                    (None, Some(dur)) => {
                        segment.set_start(gst::format::ClockTime::ZERO);
                        segment.set_stop(dur);
                    },
                    (None, None) => unreachable!(),
                };
                Some(segment)
            },
        };


        let sink_thread_data = Arc::new(
            (Mutex::new(PictureStreamSinkThreadData { per_n_frames: None, n_pulled_frames: 0, error_message: None, new_sample: None, eos: false, dead: false }), Condvar::new())
        );
        Self::setup_sink_thread(appsink, &bus, segment.clone(), opts.fps, sink_thread_data.clone());

        // PictureStreamData drop reader thread using RAII pattern
        let data = PictureStreamData {
            pipeline, _bus: bus, sink_thread_data, reader_thread_data,
        };

        // Wait for all element connected for seek;
        let state_changed = data.pipeline.set_state(gst::State::Paused)?;
        if state_changed == gst::StateChangeSuccess::Async {
            let (result, state, pending_state) = data.pipeline.state(None);
            let state_changed = result?;
            assert_eq!(state_changed, gst::StateChangeSuccess::Success);
            assert_eq!(pending_state, gst::State::VoidPending);
            assert_eq!(state, gst::State::Paused);
        }

        if let Some(segment) = segment {
            let start = segment.start();
            let stop = segment.stop();

            let gst::GenericFormattedValue::Time(Some(start)) = start else { unreachable!() };

            let (stop_type, stop) = if stop.is_none() {
                (gst::SeekType::None, gst::ClockTime::ZERO)
            } else {
                let gst::GenericFormattedValue::Time(Some(stop)) = stop else { unreachable!() };
                (gst::SeekType::Set, stop)
            };

            data.pipeline.seek(
                1.0,
                gst::SeekFlags::FLUSH | gst::SeekFlags::ACCURATE,
                gst::SeekType::Set, start, stop_type, stop,
            )?;
        };

        data.pipeline.set_state(gst::State::Playing)?;

        Ok(Self {
            data: Some(data),
        })
    }

    fn setup_reader_thread(appsrc: gst_app::AppSrc, data: Arc<(Mutex<PictureStreamReaderThreadData<R>>, Condvar)>) {
        let data_for_need_data = Arc::clone(&data);
        let data_for_enough_data = Arc::clone(&data);
        let data_for_seek_data = Arc::clone(&data);

        let appsrc_weak = appsrc.downgrade();
        thread::spawn(move || {
            // TODO error handling
            let Some(appsrc) = appsrc_weak.upgrade() else {
                return;
            };
            'thread_loop: loop {
                {
                    let (data, cvar) = &*data;
                    let mut data = data.lock().unwrap();
                    loop {
                        if data.dead {
                            break 'thread_loop;
                        }
                        if data.needs_data && !data.no_more_data {
                            break;
                        }
                        data = cvar.wait(data).expect("deny poisoned lock");
                    }
                }

                let size = 4096;
                let mut buffer = gst::Buffer::with_size(size).unwrap();

                let buffer_ref = buffer.get_mut().unwrap();
                let size = {
                    let mut buffer_map = buffer_ref.map_writable().unwrap();
                    let buffer_slice = buffer_map.as_mut_slice();

                    let (data, _) = &*data;
                    let mut data = data.lock().unwrap();
                    let size = data.read.read(buffer_slice).unwrap();
                    size
                };
                if size == 0 {
                    appsrc.end_of_stream().unwrap();
                    {
                        let (data, cvar) = &*data;
                        let mut data = data.lock().unwrap();
                        data.no_more_data = true;
                        cvar.notify_all();
                    }
                } else {
                    buffer_ref.set_size(size);
                    match appsrc.push_buffer(buffer) {
                        Ok(_) => (),
                        Err(gst::FlowError::Flushing) => (),
                        Err(err) => panic!("Failed to push buffer: {:?}", err),
                    }
                };
            };
        });

        let callbacks = gst_app::AppSrcCallbacks::builder()
            .need_data(move |_appsrc, _size| {
                let (data, cvar) = &*data_for_need_data;
                let mut data = data.lock().unwrap();
                data.needs_data = true;
                cvar.notify_all();
            })
            .enough_data(move |_appsrc| {
                let (data, cvar) = &*data_for_enough_data;
                let mut data = data.lock().unwrap();
                data.needs_data = false;
                cvar.notify_all();
            })
            .seek_data(move |_appsrc, offset| {
                let (data, cvar) = &*data_for_seek_data;
                let mut data = data.lock().unwrap();
                let pos = match data.read.seek(io::SeekFrom::Start(offset)) {
                    Err(_) => return false,
                    Ok(pos) => pos,
                };
                data.no_more_data = false;
                cvar.notify_all();
                pos == offset
            })
            .build();
        appsrc.set_callbacks(callbacks);
    }

    // seek's stop parameter sometimes ignored by demuxer, so we manually do dropping samples out of segment.
    fn setup_sink_thread(appsink: gst_app::AppSink, bus: &gst::Bus, segment: Option<gst::Segment>, fps: Option<usize>, data: Arc<(Mutex<PictureStreamSinkThreadData>, Condvar)>) {
        let data_weak_for_state_changed = Arc::downgrade(&data);
        let data_weak_for_error = Arc::downgrade(&data);
        bus.enable_sync_message_emission();

        if let Some(fps) = fps {
            let appsink_weak = appsink.downgrade();
            bus.connect_sync_message(Some("state-changed"), move |_bus, message| {
                let Some(data) = data_weak_for_state_changed.upgrade() else {
                    return;
                };
                let Some(appsink) = appsink_weak.upgrade() else {
                    return;
                };
                if message.src().unwrap() != appsink.upcast_ref::<gst::Object>() {
                    return;
                };
                let gst::MessageView::StateChanged(state_changed) = message.view() else {
                    return;
                };
                if state_changed.current() != gst::State::Paused {
                    return;
                };
                if state_changed.pending() != gst::State::VoidPending {
                    return;
                };

                let (data, cvar) = &*data;
                let mut data = data.lock().unwrap();

                // TODO error handling
                let sink_caps = appsink.static_pad("sink").unwrap().caps().unwrap();
                let frame_rate = sink_caps.structure(0).unwrap().value("framerate").unwrap();
                let frame_rate: gst::Fraction = unsafe { glib::value::FromValue::from_value(frame_rate) };
                let numer = from_c_uint(frame_rate.numer()).unwrap();
                let denom = from_c_uint(frame_rate.denom()).unwrap();
                let per_n_frames = usize::max(numer.div_ceil(fps * denom), 1);

                if let Some(current_per_n_frames) = data.per_n_frames {
                    if current_per_n_frames != per_n_frames {
                        data.error_message = Some(format!("Frame rate change not supported: {} -> {}", current_per_n_frames, per_n_frames));
                    }
                } else {
                    data.per_n_frames = Some(per_n_frames);
                };

                cvar.notify_all();
            });
        }

        bus.connect_sync_message(Some("error"), move |_bus, message| {
            let Some(data) = data_weak_for_error.upgrade() else {
                return;
            };
            let gst::MessageView::Error(err) = message.view() else {
                return;
            };

            let (data, cvar) = &*data;
            let mut data = data.lock().unwrap();
            data.error_message = Some(format!("{}", err));
            cvar.notify_all();
        });

        let data_weak_for_eos = Arc::downgrade(&data);
        let data_weak_for_new_sample = Arc::downgrade(&data);
        let callbacks = gst_app::AppSinkCallbacks::builder()
            .eos(move |_| {
                let Some(data) = data_weak_for_eos.upgrade() else {
                    return;
                };
                let (data, cvar) = &*data;
                let mut data = data.lock().unwrap();
                while !data.dead && !data.eos && data.new_sample.is_some() {
                    data = cvar.wait(data).expect("deny poisoned lock");
                }
                data.eos = true;
                cvar.notify_all();
            })
            .new_sample(move |appsink| {
                let Some(data) = data_weak_for_new_sample.upgrade() else {
                    return Ok(gst::FlowSuccess::Ok);
                };
                let (data, cvar) = &*data;
                let mut data = data.lock().unwrap();
                while !data.dead && !data.eos && data.new_sample.is_some() {
                    data = cvar.wait(data).expect("deny poisoned lock");
                }
                if data.dead || data.eos {
                    return Err(gst::FlowError::Eos);
                };
                let Some(new_sample) = appsink.try_pull_sample(gst::ClockTime::ZERO) else {
                    return Ok(gst::FlowSuccess::Ok);
                };

                let new_sample = if let Some(segment) = &segment {
                    let Some(buffer) = new_sample.buffer() else {
                        return Err(gst::FlowError::NotSupported)
                    };
                    let Some(pts) = buffer.pts() else {
                        return Err(gst::FlowError::NotSupported)
                    };

                    let gst::GenericFormattedValue::Time(Some(start)) = segment.start() else { unreachable!() };

                    if start <= pts {
                        if let gst::GenericFormattedValue::Time(Some(stop)) = segment.stop() {
                            if pts < stop {
                                Some(new_sample)
                            } else {
                                data.eos = true;
                                None
                            }
                        } else {
                            Some(new_sample)
                        }
                    } else {
                        None
                    }
                } else {
                    Some(new_sample)
                };

                if let Some(new_sample) = new_sample {
                    if let Some(per_n_frames) = data.per_n_frames {
                        if data.n_pulled_frames % per_n_frames == 0 {
                            data.new_sample = Some(new_sample);
                        }
                    } else {
                        data.new_sample = Some(new_sample);
                    }
                    data.n_pulled_frames += 1;
                }

                cvar.notify_all();
                Ok(gst::FlowSuccess::Ok)
            })
            .build();
        appsink.set_callbacks(callbacks);
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
                    // just connect
                    src_pad.link(&appsink.static_pad("sink").expect("must have a static pad")).expect("must be able to connect");
                } else {
                    // insert videoconvert elemeent
                    let videoconvert = gst::ElementFactory::make("videoconvert").build().expect("Gstreamer videoconvert element must be installed");
                    pipeline.add(&videoconvert).expect("videoconvert must be abled to be added to pipeline");
                    gst::Element::link_many([&videoconvert, &appsink.upcast_ref()]).expect("videoconvert must be connected with any raw video caps");
                    videoconvert.sync_state_with_parent().expect("videoconvert must be able to be set state");
                    src_pad.link(&videoconvert.static_pad("sink").expect("must have a static pad")).expect("must be able to connect");
                }
            })
        });
    }

    pub fn duration(&self) -> Option<gst::ClockTime> {
        let Some(ref data) = self.data else {
            return None;
        };
        let Some(time) = data.pipeline.query_duration::<gst::ClockTime>() else {
            return None;
        };
        Some(time)
    }

    pub fn position(&self) -> Option<gst::ClockTime> {
        let Some(ref data) = self.data else {
            return None;
        };
        let Some(time) = data.pipeline.query_position::<gst::ClockTime>() else {
            return None;
        };
        Some(time)
    }
}

impl<R: io::Read + io::Seek + Send + Sync + 'static> Drop for PictureStream<R> {
    fn drop(&mut self) {
        self.data = None;
    }
}

fn from_c_uint<T: TryInto<usize, Error = TryFromIntError>>(u: T) -> Result<usize, Error> {
    match u.try_into() {
        Ok(u) => Ok(u),
        Err(err) => Err(Error::FailedConversionInteger(format!("{:?}", err))),
    }
}

