use std::{io::{self, BufReader}, fs::File, path::Path};
use crate::{Error, PixelFormat, Picture};
use y4m;

#[derive(Debug)]
struct Component<'data> {
    data: &'data [u8],
    width: usize,
    height: usize,
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
    fn new(colorspace: y4m::Colorspace, width: usize, height: usize, frame: &'data y4m::Frame<'data>) -> Result<Self, Error> {
        let (pixel_format, depth, (width_subsampling, height_subsampling)): (PixelFormat, usize, (usize, usize)) = match colorspace {
            y4m::Colorspace::Cmono => (PixelFormat::Yuv400p, 8, (0, 0)),
            y4m::Colorspace::Cmono12 => (PixelFormat::Yuv400p, 12, (0, 0)),
            y4m::Colorspace::C420 => (PixelFormat::Yuv420p, 8, (2, 2)),
            y4m::Colorspace::C420p10 => (PixelFormat::Yuv420p, 10, (2, 2)),
            y4m::Colorspace::C420p12 => (PixelFormat::Yuv420p, 12, (2, 2)),
            y4m::Colorspace::C420jpeg => (PixelFormat::Yuv420p, 8, (2, 2)),
            y4m::Colorspace::C420paldv => (PixelFormat::Yuv420p, 8, (2, 2)),
            y4m::Colorspace::C420mpeg2 => (PixelFormat::Yuv420p, 8, (2, 2)),
            y4m::Colorspace::C422 => (PixelFormat::Yuv422p, 12, (2, 1)),
            y4m::Colorspace::C422p10 => (PixelFormat::Yuv422p, 10, (2, 1)),
            y4m::Colorspace::C422p12 => (PixelFormat::Yuv422p, 12, (2, 1)),
            y4m::Colorspace::C444 => (PixelFormat::Yuv444p, 8, (1, 1)),
            y4m::Colorspace::C444p10 => (PixelFormat::Yuv444p, 10, (1, 1)),
            y4m::Colorspace::C444p12 => (PixelFormat::Yuv444p, 12, (1, 1)),
            _ => return Err(Error::InvalidArgument)
        };

        let y_data = frame.get_y_plane();
        let pixel_stride = depth.div_ceil(8);

        assert_eq!(y_data.len(), width * height * pixel_stride);

        let y_component = Component { data: y_data, width, height };

        let (u_component, v_component) = if pixel_format == PixelFormat::Yuv400p {
            (None, None)
        } else {
            let chroma_width = width.div_ceil(width_subsampling);
            let chroma_height = height.div_ceil(height_subsampling);

            let u_data = frame.get_u_plane();
            assert_eq!(u_data.len(), chroma_width * chroma_height * pixel_stride);

            let v_data = frame.get_v_plane();
            assert_eq!(v_data.len(), chroma_width * chroma_height * pixel_stride);

            (Some(Component { data: u_data, width: chroma_width, height: chroma_height }), Some(Component { data: v_data, width: chroma_width, height: chroma_height }))
        };

        Ok(Self { width, height, pixel_format, depth, y_component, u_component, v_component })
    }
}

pub struct PictureStream<R: io::Read> {
    dec: y4m::Decoder<R>,
    colorspace: y4m::Colorspace,
    width: usize,
    height: usize,
}

impl<R: io::Read> crate::PictureStream for PictureStream<R> {
    fn next_pic(&mut self) -> Option<Result<Picture, Error>> {
        match self.dec.read_frame() {
            Ok(y4m_frame) => {
                match Frame::new(self.colorspace, self.width, self.height, &y4m_frame) {
                    Ok(frame) => Some(Picture::new(&frame)),
                    Err(err) => Some(Err(err)),
                }
            },
            Err(y4m::Error::EOF) => None,
            Err(err) => Some(Err(Error::from(err))),
        }
    }
}

impl PictureStream<BufReader<File>> {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        Self::new(BufReader::new(File::open(path).map_err(Error::IoError)?))
    }
}

impl<R: io::Read> PictureStream<R> {
    pub fn new(read: R) -> Result<Self, Error> {
        let dec = y4m::decode(read).map_err(Error::from)?;
        Ok(Self::from_y4m_dec(dec))
    }

    fn from_y4m_dec(dec: y4m::Decoder<R>) -> Self {
        let colorspace = dec.get_colorspace();
        let width = dec.get_width();
        let height = dec.get_height();
        Self { dec, colorspace, width, height }
    }
}

impl From<y4m::Error> for Error {
    fn from(err: y4m::Error) -> Self {
        match err {
            y4m::Error::BadInput | y4m::Error::UnknownColorspace => Error::InvalidArgument,
            y4m::Error::ParseError(err) => Error::InvalidVideoFrameFormat(err.to_string()),
            y4m::Error::IoError(err) => Error::IoError(err),
            y4m::Error::OutOfMemory => Error::OutOfMemory,
            y4m::Error::EOF => unreachable!(),
        }
    }
}

