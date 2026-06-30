#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// Corresponds to sensor_msgs__msg__BatteryState

// This struct is not documented.
#[allow(missing_docs)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BatteryState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    /// Voltage in Volts (Mandatory)
    pub voltage: f32,

    /// Temperature in Degrees Celsius (If unmeasured NaN)
    pub temperature: f32,

    /// Negative when discharging (A)  (If unmeasured NaN)
    pub current: f32,

    /// Current charge in Ah  (If unmeasured NaN)
    pub charge: f32,

    /// Capacity in Ah (last full capacity)  (If unmeasured NaN)
    pub capacity: f32,

    /// Capacity in Ah (design capacity)  (If unmeasured NaN)
    pub design_capacity: f32,

    /// Charge percentage on 0 to 1 range  (If unmeasured NaN)
    pub percentage: f32,

    /// The charging status as reported. Values defined above
    pub power_supply_status: u8,

    /// The battery health metric. Values defined above
    pub power_supply_health: u8,

    /// The battery chemistry. Values defined above
    pub power_supply_technology: u8,

    /// True if the battery is present
    pub present: bool,

    /// An array of individual cell voltages for each cell in the pack
    /// If individual voltages unknown but number of cells known set each to NaN
    pub cell_voltage: Vec<f32>,

    /// An array of individual cell temperatures for each cell in the pack
    /// If individual temperatures unknown but number of cells known set each to NaN
    pub cell_temperature: Vec<f32>,

    /// The location into which the battery is inserted. (slot number or plug)
    pub location: std::string::String,

    /// The best approximation of the battery serial number
    pub serial_number: std::string::String,
}

impl BatteryState {
    /// Constants are chosen to match the enums in the linux kernel
    /// defined in include/linux/power_supply.h as of version 3.7
    /// The one difference is for style reasons the constants are
    /// all uppercase not mixed case.
    /// Power supply status constants
    pub const POWER_SUPPLY_STATUS_UNKNOWN: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_STATUS_CHARGING: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_STATUS_DISCHARGING: u8 = 2;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_STATUS_NOT_CHARGING: u8 = 3;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_STATUS_FULL: u8 = 4;

    /// Power supply health constants
    pub const POWER_SUPPLY_HEALTH_UNKNOWN: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_GOOD: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_OVERHEAT: u8 = 2;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_DEAD: u8 = 3;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_OVERVOLTAGE: u8 = 4;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_UNSPEC_FAILURE: u8 = 5;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_COLD: u8 = 6;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_WATCHDOG_TIMER_EXPIRE: u8 = 7;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_HEALTH_SAFETY_TIMER_EXPIRE: u8 = 8;

    /// Power supply technology (chemistry) constants
    pub const POWER_SUPPLY_TECHNOLOGY_UNKNOWN: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_NIMH: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_LION: u8 = 2;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_LIPO: u8 = 3;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_LIFE: u8 = 4;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_NICD: u8 = 5;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const POWER_SUPPLY_TECHNOLOGY_LIMN: u8 = 6;
}

impl Default for BatteryState {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::BatteryState::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for BatteryState {
    type RmwMsg = super::msg::rmw::BatteryState;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                voltage: msg.voltage,
                temperature: msg.temperature,
                current: msg.current,
                charge: msg.charge,
                capacity: msg.capacity,
                design_capacity: msg.design_capacity,
                percentage: msg.percentage,
                power_supply_status: msg.power_supply_status,
                power_supply_health: msg.power_supply_health,
                power_supply_technology: msg.power_supply_technology,
                present: msg.present,
                cell_voltage: msg.cell_voltage.into(),
                cell_temperature: msg.cell_temperature.into(),
                location: msg.location.as_str().into(),
                serial_number: msg.serial_number.as_str().into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                voltage: msg.voltage,
                temperature: msg.temperature,
                current: msg.current,
                charge: msg.charge,
                capacity: msg.capacity,
                design_capacity: msg.design_capacity,
                percentage: msg.percentage,
                power_supply_status: msg.power_supply_status,
                power_supply_health: msg.power_supply_health,
                power_supply_technology: msg.power_supply_technology,
                present: msg.present,
                cell_voltage: msg.cell_voltage.as_slice().into(),
                cell_temperature: msg.cell_temperature.as_slice().into(),
                location: msg.location.as_str().into(),
                serial_number: msg.serial_number.as_str().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            voltage: msg.voltage,
            temperature: msg.temperature,
            current: msg.current,
            charge: msg.charge,
            capacity: msg.capacity,
            design_capacity: msg.design_capacity,
            percentage: msg.percentage,
            power_supply_status: msg.power_supply_status,
            power_supply_health: msg.power_supply_health,
            power_supply_technology: msg.power_supply_technology,
            present: msg.present,
            cell_voltage: msg.cell_voltage.into_iter().collect(),
            cell_temperature: msg.cell_temperature.into_iter().collect(),
            location: msg.location.to_string(),
            serial_number: msg.serial_number.to_string(),
        }
    }
}

// Corresponds to sensor_msgs__msg__CameraInfo
/// This message defines meta information for a camera. It should be in a
/// camera namespace on topic "camera_info" and accompanied by up to five
/// image topics named:
///
///   image_raw - raw data from the camera driver, possibly Bayer encoded
///   image            - monochrome, distorted
///   image_color      - color, distorted
///   image_rect       - monochrome, rectified
///   image_rect_color - color, rectified
///
/// The image_pipeline contains packages (image_proc, stereo_image_proc)
/// for producing the four processed image topics from image_raw and
/// camera_info. The meaning of the camera parameters are described in
/// detail at http://www.ros.org/wiki/image_pipeline/CameraInfo.
///
/// The image_geometry package provides a user-friendly interface to
/// common operations using this meta information. If you want to, e.g.,
/// project a 3d point into image coordinates, we strongly recommend
/// using image_geometry.
///
/// If the camera is uncalibrated, the matrices D, K, R, P should be left
/// zeroed out. In particular, clients may assume that K == 0.0
/// indicates an uncalibrated camera.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CameraInfo {
    ///                     Image acquisition info                          #
    ///
    /// Time of image acquisition, camera coordinate frame ID
    /// Header timestamp should be acquisition time of image
    /// Header frame_id should be optical frame of camera
    /// origin of frame should be optical center of camera
    /// +x should point to the right in the image
    /// +y should point down in the image
    /// +z should point into the plane of the image
    pub header: std_msgs::msg::Header,

    ///                      Calibration Parameters                         #
    ///
    /// These are fixed during camera calibration. Their values will be the #
    /// same in all messages until the camera is recalibrated. Note that    #
    /// self-calibrating systems may "recalibrate" frequently.              #
    ///                                                                     #
    /// The internal parameters can be used to warp a raw (distorted) image #
    /// to:                                                                 #
    ///   1. An undistorted image (requires D and K)                        #
    ///   2. A rectified image (requires D, K, R)                           #
    /// The projection matrix P projects 3D points into the rectified image.#
    ///
    /// The image dimensions with which the camera was calibrated.
    /// Normally this will be the full camera resolution in pixels.
    pub height: u32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub width: u32,

    /// The distortion model used. Supported models are listed in
    /// sensor_msgs/distortion_models.hpp. For most cameras, "plumb_bob" - a
    /// simple model of radial and tangential distortion - is sufficent.
    pub distortion_model: std::string::String,

    /// The distortion parameters, size depending on the distortion model.
    /// For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
    pub d: Vec<f64>,

    /// Intrinsic camera matrix for the raw (distorted) images.
    ///     [fx  0 cx]
    /// K = [ 0 fy cy]
    ///     [ 0  0  1]
    /// Projects 3D points in the camera coordinate frame to 2D pixel
    /// coordinates using the focal lengths (fx, fy) and principal point
    /// (cx, cy).
    /// 3x3 row-major matrix
    pub k: [f64; 9],

    /// Rectification matrix (stereo cameras only)
    /// A rotation matrix aligning the camera coordinate system to the ideal
    /// stereo image plane so that epipolar lines in both stereo images are
    /// parallel.
    /// 3x3 row-major matrix
    pub r: [f64; 9],

    /// Projection/camera matrix
    ///     [fx'  0  cx' Tx]
    /// P = [ 0  fy' cy' Ty]
    ///     [ 0   0   1   0]
    /// By convention, this matrix specifies the intrinsic (camera) matrix
    ///  of the processed (rectified) image. That is, the left 3x3 portion
    ///  is the normal camera intrinsic matrix for the rectified image.
    /// It projects 3D points in the camera coordinate frame to 2D pixel
    ///  coordinates using the focal lengths (fx', fy') and principal point
    ///  (cx', cy') - these may differ from the values in K.
    /// For monocular cameras, Tx = Ty = 0. Normally, monocular cameras will
    ///  also have R = the identity and P[1:3,1:3] = K.
    /// For a stereo pair, the fourth column [Tx Ty 0]' is related to the
    ///  position of the optical center of the second camera in the first
    ///  camera's frame. We assume Tz = 0 so both cameras are in the same
    ///  stereo image plane. The first camera always has Tx = Ty = 0. For
    ///  the right (second) camera of a horizontal stereo pair, Ty = 0 and
    ///  Tx = -fx' * B, where B is the baseline between the cameras.
    /// Given a 3D point [X Y Z]', the projection (x, y) of the point onto
    ///  the rectified image is given by:
    ///  [u v w]' = P * [X Y Z 1]'
    ///         x = u / w
    ///         y = v / w
    ///  This holds for both images of a stereo pair.
    /// 3x4 row-major matrix
    pub p: [f64; 12],

    ///                      Operational Parameters                         #
    ///
    /// These define the image region actually captured by the camera       #
    /// driver. Although they affect the geometry of the output image, they #
    /// may be changed freely without recalibrating the camera.             #
    ///
    /// Binning refers here to any camera setting which combines rectangular
    ///  neighborhoods of pixels into larger "super-pixels." It reduces the
    ///  resolution of the output image to
    ///  (width / binning_x) x (height / binning_y).
    /// The default values binning_x = binning_y = 0 is considered the same
    ///  as binning_x = binning_y = 1 (no subsampling).
    pub binning_x: u32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub binning_y: u32,

    /// Region of interest (subwindow of full camera resolution), given in
    ///  full resolution (unbinned) image coordinates. A particular ROI
    ///  always denotes the same window of pixels on the camera sensor,
    ///  regardless of binning settings.
    /// The default setting of roi (all values 0) is considered the same as
    ///  full resolution (roi.width = width, roi.height = height).
    pub roi: super::msg::RegionOfInterest,
}

impl Default for CameraInfo {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::CameraInfo::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for CameraInfo {
    type RmwMsg = super::msg::rmw::CameraInfo;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                distortion_model: msg.distortion_model.as_str().into(),
                d: msg.d.into(),
                k: msg.k,
                r: msg.r,
                p: msg.p,
                binning_x: msg.binning_x,
                binning_y: msg.binning_y,
                roi: super::msg::RegionOfInterest::into_rmw_message(std::borrow::Cow::Owned(
                    msg.roi,
                ))
                .into_owned(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                distortion_model: msg.distortion_model.as_str().into(),
                d: msg.d.as_slice().into(),
                k: msg.k,
                r: msg.r,
                p: msg.p,
                binning_x: msg.binning_x,
                binning_y: msg.binning_y,
                roi: super::msg::RegionOfInterest::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.roi,
                ))
                .into_owned(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            height: msg.height,
            width: msg.width,
            distortion_model: msg.distortion_model.to_string(),
            d: msg.d.into_iter().collect(),
            k: msg.k,
            r: msg.r,
            p: msg.p,
            binning_x: msg.binning_x,
            binning_y: msg.binning_y,
            roi: super::msg::RegionOfInterest::from_rmw_message(msg.roi),
        }
    }
}

// Corresponds to sensor_msgs__msg__ChannelFloat32
/// This message is used by the PointCloud message to hold optional data
/// associated with each point in the cloud. The length of the values
/// array should be the same as the length of the points array in the
/// PointCloud, and each value should be associated with the corresponding
/// point.
///
/// Channel names in existing practice include:
///   "u", "v" - row and column (respectively) in the left stereo image.
///              This is opposite to usual conventions but remains for
///              historical reasons. The newer PointCloud2 message has no
///              such problem.
///   "rgb" - For point clouds produced by color stereo cameras. uint8
///           (R,G,B) values packed into the least significant 24 bits,
///           in order.
///   "intensity" - laser or pixel intensity.
///   "distance"

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ChannelFloat32 {
    /// The channel name should give semantics of the channel (e.g.
    /// "intensity" instead of "value").
    pub name: std::string::String,

    /// The values array should be 1-1 with the elements of the associated
    /// PointCloud.
    pub values: Vec<f32>,
}

impl Default for ChannelFloat32 {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::ChannelFloat32::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for ChannelFloat32 {
    type RmwMsg = super::msg::rmw::ChannelFloat32;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                name: msg.name.as_str().into(),
                values: msg.values.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                name: msg.name.as_str().into(),
                values: msg.values.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            name: msg.name.to_string(),
            values: msg.values.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__CompressedImage
/// This message contains a compressed image.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CompressedImage {
    /// Header timestamp should be acquisition time of image
    /// Header frame_id should be optical frame of camera
    /// origin of frame should be optical center of cameara
    /// +x should point to the right in the image
    /// +y should point down in the image
    /// +z should point into to plane of the image
    pub header: std_msgs::msg::Header,

    /// Specifies the format of the data
    ///   Acceptable values:
    ///     jpeg, png, tiff
    pub format: std::string::String,

    /// Compressed image buffer
    pub data: Vec<u8>,
}

impl Default for CompressedImage {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::CompressedImage::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for CompressedImage {
    type RmwMsg = super::msg::rmw::CompressedImage;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                format: msg.format.as_str().into(),
                data: msg.data.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                format: msg.format.as_str().into(),
                data: msg.data.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            format: msg.format.to_string(),
            data: msg.data.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__FluidPressure
/// Single pressure reading.  This message is appropriate for measuring the
/// pressure inside of a fluid (air, water, etc).  This also includes
/// atmospheric or barometric pressure.
///
/// This message is not appropriate for force/pressure contact sensors.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FluidPressure {
    /// timestamp of the measurement
    /// frame_id is the location of the pressure sensor
    pub header: std_msgs::msg::Header,

    /// Absolute pressure reading in Pascals.
    pub fluid_pressure: f64,

    /// 0 is interpreted as variance unknown
    pub variance: f64,
}

impl Default for FluidPressure {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::FluidPressure::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for FluidPressure {
    type RmwMsg = super::msg::rmw::FluidPressure;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                fluid_pressure: msg.fluid_pressure,
                variance: msg.variance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                fluid_pressure: msg.fluid_pressure,
                variance: msg.variance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            fluid_pressure: msg.fluid_pressure,
            variance: msg.variance,
        }
    }
}

// Corresponds to sensor_msgs__msg__Illuminance
/// Single photometric illuminance measurement.  Light should be assumed to be
/// measured along the sensor's x-axis (the area of detection is the y-z plane).
/// The illuminance should have a 0 or positive value and be received with
/// the sensor's +X axis pointing toward the light source.
///
/// Photometric illuminance is the measure of the human eye's sensitivity of the
/// intensity of light encountering or passing through a surface.
///
/// All other Photometric and Radiometric measurements should not use this message.
/// This message cannot represent:
///  - Luminous intensity (candela/light source output)
///  - Luminance (nits/light output per area)
///  - Irradiance (watt/area), etc.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Illuminance {
    /// timestamp is the time the illuminance was measured
    /// frame_id is the location and direction of the reading
    pub header: std_msgs::msg::Header,

    /// Measurement of the Photometric Illuminance in Lux.
    pub illuminance: f64,

    /// 0 is interpreted as variance unknown
    pub variance: f64,
}

impl Default for Illuminance {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::Illuminance::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for Illuminance {
    type RmwMsg = super::msg::rmw::Illuminance;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                illuminance: msg.illuminance,
                variance: msg.variance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                illuminance: msg.illuminance,
                variance: msg.variance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            illuminance: msg.illuminance,
            variance: msg.variance,
        }
    }
}

// Corresponds to sensor_msgs__msg__Image
/// This message contains an uncompressed image
/// (0, 0) is at top-left corner of image

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Image {
    /// Header timestamp should be acquisition time of image
    /// Header frame_id should be optical frame of camera
    /// origin of frame should be optical center of cameara
    /// +x should point to the right in the image
    /// +y should point down in the image
    /// +z should point into to plane of the image
    /// If the frame_id here and the frame_id of the CameraInfo
    /// message associated with the image conflict
    /// the behavior is undefined
    pub header: std_msgs::msg::Header,

    /// image height, that is, number of rows
    pub height: u32,

    /// image width, that is, number of columns
    pub width: u32,

    /// The legal values for encoding are in file src/image_encodings.cpp
    /// If you want to standardize a new string format, join
    /// ros-users@lists.ros.org and send an email proposing a new encoding.
    /// Encoding of pixels -- channel meaning, ordering, size
    /// taken from the list of strings in include/sensor_msgs/image_encodings.hpp
    pub encoding: std::string::String,

    /// is this data bigendian?
    pub is_bigendian: u8,

    /// Full row length in bytes
    pub step: u32,

    /// actual matrix data, size is (step * rows)
    pub data: Vec<u8>,
}

impl Default for Image {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Image::default())
    }
}

impl rosidl_runtime_rs::Message for Image {
    type RmwMsg = super::msg::rmw::Image;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                encoding: msg.encoding.as_str().into(),
                is_bigendian: msg.is_bigendian,
                step: msg.step,
                data: msg.data.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                encoding: msg.encoding.as_str().into(),
                is_bigendian: msg.is_bigendian,
                step: msg.step,
                data: msg.data.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            height: msg.height,
            width: msg.width,
            encoding: msg.encoding.to_string(),
            is_bigendian: msg.is_bigendian,
            step: msg.step,
            data: msg.data.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__Imu
/// This is a message to hold data from an IMU (Inertial Measurement Unit)
///
/// Accelerations should be in m/s^2 (not in g's), and rotational velocity should be in rad/sec
///
/// If the covariance of the measurement is known, it should be filled in (if all you know is the
/// variance of each measurement, e.g. from the datasheet, just put those along the diagonal)
/// A covariance matrix of all zeros will be interpreted as "covariance unknown", and to use the
/// data a covariance will have to be assumed or gotten from some other source
///
/// If you have no estimate for one of the data elements (e.g. your IMU doesn't produce an
/// orientation estimate), please set element 0 of the associated covariance matrix to -1
/// If you are interpreting this message, please check for a value of -1 in the first element of each
/// covariance matrix, and disregard the associated estimate.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Imu {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub orientation: geometry_msgs::msg::Quaternion,

    /// Row major about x, y, z axes
    pub orientation_covariance: [f64; 9],

    // This member is not documented.
    #[allow(missing_docs)]
    pub angular_velocity: geometry_msgs::msg::Vector3,

    /// Row major about x, y, z axes
    pub angular_velocity_covariance: [f64; 9],

    // This member is not documented.
    #[allow(missing_docs)]
    pub linear_acceleration: geometry_msgs::msg::Vector3,

    /// Row major x, y z
    pub linear_acceleration_covariance: [f64; 9],
}

impl Default for Imu {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Imu::default())
    }
}

impl rosidl_runtime_rs::Message for Imu {
    type RmwMsg = super::msg::rmw::Imu;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                orientation: geometry_msgs::msg::Quaternion::into_rmw_message(
                    std::borrow::Cow::Owned(msg.orientation),
                )
                .into_owned(),
                orientation_covariance: msg.orientation_covariance,
                angular_velocity: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Owned(msg.angular_velocity),
                )
                .into_owned(),
                angular_velocity_covariance: msg.angular_velocity_covariance,
                linear_acceleration: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Owned(msg.linear_acceleration),
                )
                .into_owned(),
                linear_acceleration_covariance: msg.linear_acceleration_covariance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                orientation: geometry_msgs::msg::Quaternion::into_rmw_message(
                    std::borrow::Cow::Borrowed(&msg.orientation),
                )
                .into_owned(),
                orientation_covariance: msg.orientation_covariance,
                angular_velocity: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Borrowed(&msg.angular_velocity),
                )
                .into_owned(),
                angular_velocity_covariance: msg.angular_velocity_covariance,
                linear_acceleration: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Borrowed(&msg.linear_acceleration),
                )
                .into_owned(),
                linear_acceleration_covariance: msg.linear_acceleration_covariance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            orientation: geometry_msgs::msg::Quaternion::from_rmw_message(msg.orientation),
            orientation_covariance: msg.orientation_covariance,
            angular_velocity: geometry_msgs::msg::Vector3::from_rmw_message(msg.angular_velocity),
            angular_velocity_covariance: msg.angular_velocity_covariance,
            linear_acceleration: geometry_msgs::msg::Vector3::from_rmw_message(
                msg.linear_acceleration,
            ),
            linear_acceleration_covariance: msg.linear_acceleration_covariance,
        }
    }
}

// Corresponds to sensor_msgs__msg__JointState
/// This is a message that holds data to describe the state of a set of torque controlled joints.
///
/// The state of each joint (revolute or prismatic) is defined by:
///  * the position of the joint (rad or m),
///  * the velocity of the joint (rad/s or m/s) and
///  * the effort that is applied in the joint (Nm or N).
///
/// Each joint is uniquely identified by its name
/// The header specifies the time at which the joint states were recorded. All the joint states
/// in one message have to be recorded at the same time.
///
/// This message consists of a multiple arrays, one for each part of the joint state.
/// The goal is to make each of the fields optional. When e.g. your joints have no
/// effort associated with them, you can leave the effort array empty.
///
/// All arrays in this message should have the same size, or be empty.
/// This is the only way to uniquely associate the joint name with the correct
/// states.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: Vec<std::string::String>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub position: Vec<f64>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: Vec<f64>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub effort: Vec<f64>,
}

impl Default for JointState {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::JointState::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for JointState {
    type RmwMsg = super::msg::rmw::JointState;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                name: msg
                    .name
                    .into_iter()
                    .map(|elem| elem.as_str().into())
                    .collect(),
                position: msg.position.into(),
                velocity: msg.velocity.into(),
                effort: msg.effort.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                name: msg.name.iter().map(|elem| elem.as_str().into()).collect(),
                position: msg.position.as_slice().into(),
                velocity: msg.velocity.as_slice().into(),
                effort: msg.effort.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            name: msg.name.into_iter().map(|elem| elem.to_string()).collect(),
            position: msg.position.into_iter().collect(),
            velocity: msg.velocity.into_iter().collect(),
            effort: msg.effort.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__Joy
/// Reports the state of a joystick's axes and buttons.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Joy {
    /// The timestamp is the time at which data is received from the joystick.
    pub header: std_msgs::msg::Header,

    /// The axes measurements from a joystick.
    pub axes: Vec<f32>,

    /// The buttons measurements from a joystick.
    pub buttons: Vec<i32>,
}

impl Default for Joy {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Joy::default())
    }
}

impl rosidl_runtime_rs::Message for Joy {
    type RmwMsg = super::msg::rmw::Joy;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                axes: msg.axes.into(),
                buttons: msg.buttons.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                axes: msg.axes.as_slice().into(),
                buttons: msg.buttons.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            axes: msg.axes.into_iter().collect(),
            buttons: msg.buttons.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__JoyFeedback
/// Declare of the type of feedback

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JoyFeedback {
    // This member is not documented.
    #[allow(missing_docs)]
    pub type_: u8,

    /// This will hold an id number for each type of each feedback.
    /// Example, the first led would be id=0, the second would be id=1
    pub id: u8,

    /// Intensity of the feedback, from 0.0 to 1.0, inclusive.  If device is
    /// actually binary, driver should treat 0<=x<0.5 as off, 0.5<=x<=1 as on.
    pub intensity: f32,
}

impl JoyFeedback {
    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TYPE_LED: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TYPE_RUMBLE: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const TYPE_BUZZER: u8 = 2;
}

impl Default for JoyFeedback {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::JoyFeedback::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for JoyFeedback {
    type RmwMsg = super::msg::rmw::JoyFeedback;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                type_: msg.type_,
                id: msg.id,
                intensity: msg.intensity,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                type_: msg.type_,
                id: msg.id,
                intensity: msg.intensity,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            type_: msg.type_,
            id: msg.id,
            intensity: msg.intensity,
        }
    }
}

// Corresponds to sensor_msgs__msg__JoyFeedbackArray
/// This message publishes values for multiple feedback at once.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JoyFeedbackArray {
    // This member is not documented.
    #[allow(missing_docs)]
    pub array: Vec<super::msg::JoyFeedback>,
}

impl Default for JoyFeedbackArray {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::JoyFeedbackArray::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for JoyFeedbackArray {
    type RmwMsg = super::msg::rmw::JoyFeedbackArray;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                array: msg
                    .array
                    .into_iter()
                    .map(|elem| {
                        super::msg::JoyFeedback::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                array: msg
                    .array
                    .iter()
                    .map(|elem| {
                        super::msg::JoyFeedback::into_rmw_message(std::borrow::Cow::Borrowed(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            array: msg
                .array
                .into_iter()
                .map(super::msg::JoyFeedback::from_rmw_message)
                .collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__LaserEcho
/// This message is a submessage of MultiEchoLaserScan and is not intended
/// to be used separately.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LaserEcho {
    /// Multiple values of ranges or intensities.
    /// Each array represents data from the same angle increment.
    pub echoes: Vec<f32>,
}

impl Default for LaserEcho {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::LaserEcho::default())
    }
}

impl rosidl_runtime_rs::Message for LaserEcho {
    type RmwMsg = super::msg::rmw::LaserEcho;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                echoes: msg.echoes.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                echoes: msg.echoes.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            echoes: msg.echoes.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__LaserScan
/// Single scan from a planar laser range-finder
///
/// If you have another ranging device with different behavior (e.g. a sonar
/// array), please find or create a different message, since applications
/// will make fairly laser-specific assumptions about this data

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LaserScan {
    /// timestamp in the header is the acquisition time of
    /// the first ray in the scan.
    ///
    /// in frame frame_id, angles are measured around
    /// the positive Z axis (counterclockwise, if Z is up)
    /// with zero angle being forward along the x axis
    pub header: std_msgs::msg::Header,

    /// start angle of the scan
    pub angle_min: f32,

    /// end angle of the scan
    pub angle_max: f32,

    /// angular distance between measurements
    pub angle_increment: f32,

    /// time between measurements - if your scanner
    /// is moving, this will be used in interpolating position
    /// of 3d points
    pub time_increment: f32,

    /// time between scans
    pub scan_time: f32,

    /// minimum range value
    pub range_min: f32,

    /// maximum range value
    pub range_max: f32,

    /// range data
    /// (Note: values < range_min or > range_max should be discarded)
    pub ranges: Vec<f32>,

    /// intensity data.  If your
    /// device does not provide intensities, please leave
    /// the array empty.
    pub intensities: Vec<f32>,
}

impl Default for LaserScan {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::LaserScan::default())
    }
}

impl rosidl_runtime_rs::Message for LaserScan {
    type RmwMsg = super::msg::rmw::LaserScan;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                angle_min: msg.angle_min,
                angle_max: msg.angle_max,
                angle_increment: msg.angle_increment,
                time_increment: msg.time_increment,
                scan_time: msg.scan_time,
                range_min: msg.range_min,
                range_max: msg.range_max,
                ranges: msg.ranges.into(),
                intensities: msg.intensities.into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                angle_min: msg.angle_min,
                angle_max: msg.angle_max,
                angle_increment: msg.angle_increment,
                time_increment: msg.time_increment,
                scan_time: msg.scan_time,
                range_min: msg.range_min,
                range_max: msg.range_max,
                ranges: msg.ranges.as_slice().into(),
                intensities: msg.intensities.as_slice().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            angle_min: msg.angle_min,
            angle_max: msg.angle_max,
            angle_increment: msg.angle_increment,
            time_increment: msg.time_increment,
            scan_time: msg.scan_time,
            range_min: msg.range_min,
            range_max: msg.range_max,
            ranges: msg.ranges.into_iter().collect(),
            intensities: msg.intensities.into_iter().collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__MagneticField
/// Measurement of the Magnetic Field vector at a specific location.
///
/// If the covariance of the measurement is known, it should be filled in.
/// If all you know is the variance of each measurement, e.g. from the datasheet,
/// just put those along the diagonal.
/// A covariance matrix of all zeros will be interpreted as "covariance unknown",
/// and to use the data a covariance will have to be assumed or gotten from some
/// other source.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MagneticField {
    /// timestamp is the time the
    /// field was measured
    /// frame_id is the location and orientation
    /// of the field measurement
    pub header: std_msgs::msg::Header,

    /// x, y, and z components of the
    /// field vector in Tesla
    /// If your sensor does not output 3 axes,
    /// put NaNs in the components not reported.
    pub magnetic_field: geometry_msgs::msg::Vector3,

    /// Row major about x, y, z axes
    /// 0 is interpreted as variance unknown
    pub magnetic_field_covariance: [f64; 9],
}

impl Default for MagneticField {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::MagneticField::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for MagneticField {
    type RmwMsg = super::msg::rmw::MagneticField;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                magnetic_field: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Owned(msg.magnetic_field),
                )
                .into_owned(),
                magnetic_field_covariance: msg.magnetic_field_covariance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                magnetic_field: geometry_msgs::msg::Vector3::into_rmw_message(
                    std::borrow::Cow::Borrowed(&msg.magnetic_field),
                )
                .into_owned(),
                magnetic_field_covariance: msg.magnetic_field_covariance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            magnetic_field: geometry_msgs::msg::Vector3::from_rmw_message(msg.magnetic_field),
            magnetic_field_covariance: msg.magnetic_field_covariance,
        }
    }
}

// Corresponds to sensor_msgs__msg__MultiDOFJointState
/// Representation of state for joints with multiple degrees of freedom,
/// following the structure of JointState which can only represent a single degree of freedom.
///
/// It is assumed that a joint in a system corresponds to a transform that gets applied
/// along the kinematic chain. For example, a planar joint (as in URDF) is 3DOF (x, y, yaw)
/// and those 3DOF can be expressed as a transformation matrix, and that transformation
/// matrix can be converted back to (x, y, yaw)
///
/// Each joint is uniquely identified by its name
/// The header specifies the time at which the joint states were recorded. All the joint states
/// in one message have to be recorded at the same time.
///
/// This message consists of a multiple arrays, one for each part of the joint state.
/// The goal is to make each of the fields optional. When e.g. your joints have no
/// wrench associated with them, you can leave the wrench array empty.
///
/// All arrays in this message should have the same size, or be empty.
/// This is the only way to uniquely associate the joint name with the correct
/// states.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_names: Vec<std::string::String>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub transforms: Vec<geometry_msgs::msg::Transform>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: Vec<geometry_msgs::msg::Twist>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub wrench: Vec<geometry_msgs::msg::Wrench>,
}

impl Default for MultiDOFJointState {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::MultiDOFJointState::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for MultiDOFJointState {
    type RmwMsg = super::msg::rmw::MultiDOFJointState;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                joint_names: msg
                    .joint_names
                    .into_iter()
                    .map(|elem| elem.as_str().into())
                    .collect(),
                transforms: msg
                    .transforms
                    .into_iter()
                    .map(|elem| {
                        geometry_msgs::msg::Transform::into_rmw_message(std::borrow::Cow::Owned(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
                twist: msg
                    .twist
                    .into_iter()
                    .map(|elem| {
                        geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
                wrench: msg
                    .wrench
                    .into_iter()
                    .map(|elem| {
                        geometry_msgs::msg::Wrench::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                joint_names: msg
                    .joint_names
                    .iter()
                    .map(|elem| elem.as_str().into())
                    .collect(),
                transforms: msg
                    .transforms
                    .iter()
                    .map(|elem| {
                        geometry_msgs::msg::Transform::into_rmw_message(std::borrow::Cow::Borrowed(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
                twist: msg
                    .twist
                    .iter()
                    .map(|elem| {
                        geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
                wrench: msg
                    .wrench
                    .iter()
                    .map(|elem| {
                        geometry_msgs::msg::Wrench::into_rmw_message(std::borrow::Cow::Borrowed(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            joint_names: msg
                .joint_names
                .into_iter()
                .map(|elem| elem.to_string())
                .collect(),
            transforms: msg
                .transforms
                .into_iter()
                .map(geometry_msgs::msg::Transform::from_rmw_message)
                .collect(),
            twist: msg
                .twist
                .into_iter()
                .map(geometry_msgs::msg::Twist::from_rmw_message)
                .collect(),
            wrench: msg
                .wrench
                .into_iter()
                .map(geometry_msgs::msg::Wrench::from_rmw_message)
                .collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__MultiEchoLaserScan
/// Single scan from a multi-echo planar laser range-finder
///
/// If you have another ranging device with different behavior (e.g. a sonar
/// array), please find or create a different message, since applications
/// will make fairly laser-specific assumptions about this data

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiEchoLaserScan {
    /// timestamp in the header is the acquisition time of
    /// the first ray in the scan.
    ///
    /// in frame frame_id, angles are measured around
    /// the positive Z axis (counterclockwise, if Z is up)
    /// with zero angle being forward along the x axis
    pub header: std_msgs::msg::Header,

    /// start angle of the scan
    pub angle_min: f32,

    /// end angle of the scan
    pub angle_max: f32,

    /// angular distance between measurements
    pub angle_increment: f32,

    /// time between measurements - if your scanner
    /// is moving, this will be used in interpolating position
    /// of 3d points
    pub time_increment: f32,

    /// time between scans
    pub scan_time: f32,

    /// minimum range value
    pub range_min: f32,

    /// maximum range value
    pub range_max: f32,

    /// range data
    /// (Note: NaNs, values < range_min or > range_max should be discarded)
    /// +Inf measurements are out of range
    /// -Inf measurements are too close to determine exact distance.
    pub ranges: Vec<super::msg::LaserEcho>,

    /// intensity data.  If your
    /// device does not provide intensities, please leave
    /// the array empty.
    pub intensities: Vec<super::msg::LaserEcho>,
}

impl Default for MultiEchoLaserScan {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::MultiEchoLaserScan::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for MultiEchoLaserScan {
    type RmwMsg = super::msg::rmw::MultiEchoLaserScan;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                angle_min: msg.angle_min,
                angle_max: msg.angle_max,
                angle_increment: msg.angle_increment,
                time_increment: msg.time_increment,
                scan_time: msg.scan_time,
                range_min: msg.range_min,
                range_max: msg.range_max,
                ranges: msg
                    .ranges
                    .into_iter()
                    .map(|elem| {
                        super::msg::LaserEcho::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
                intensities: msg
                    .intensities
                    .into_iter()
                    .map(|elem| {
                        super::msg::LaserEcho::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                angle_min: msg.angle_min,
                angle_max: msg.angle_max,
                angle_increment: msg.angle_increment,
                time_increment: msg.time_increment,
                scan_time: msg.scan_time,
                range_min: msg.range_min,
                range_max: msg.range_max,
                ranges: msg
                    .ranges
                    .iter()
                    .map(|elem| {
                        super::msg::LaserEcho::into_rmw_message(std::borrow::Cow::Borrowed(elem))
                            .into_owned()
                    })
                    .collect(),
                intensities: msg
                    .intensities
                    .iter()
                    .map(|elem| {
                        super::msg::LaserEcho::into_rmw_message(std::borrow::Cow::Borrowed(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            angle_min: msg.angle_min,
            angle_max: msg.angle_max,
            angle_increment: msg.angle_increment,
            time_increment: msg.time_increment,
            scan_time: msg.scan_time,
            range_min: msg.range_min,
            range_max: msg.range_max,
            ranges: msg
                .ranges
                .into_iter()
                .map(super::msg::LaserEcho::from_rmw_message)
                .collect(),
            intensities: msg
                .intensities
                .into_iter()
                .map(super::msg::LaserEcho::from_rmw_message)
                .collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__NavSatFix
/// Navigation Satellite fix for any Global Navigation Satellite System
///
/// Specified using the WGS 84 reference ellipsoid

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct NavSatFix {
    /// header.stamp specifies the ROS time for this measurement (the
    ///        corresponding satellite time may be reported using the
    ///        sensor_msgs/TimeReference message).
    ///
    /// header.frame_id is the frame of reference reported by the satellite
    ///        receiver, usually the location of the antenna.  This is a
    ///        Euclidean frame relative to the vehicle, not a reference
    ///        ellipsoid.
    pub header: std_msgs::msg::Header,

    /// Satellite fix status information.
    pub status: super::msg::NavSatStatus,

    /// Latitude. Positive is north of equator; negative is south.
    pub latitude: f64,

    /// Longitude. Positive is east of prime meridian; negative is west.
    pub longitude: f64,

    /// Altitude. Positive is above the WGS 84 ellipsoid
    /// (quiet NaN if no altitude is available).
    pub altitude: f64,

    /// Position covariance defined relative to a tangential plane
    /// through the reported position. The components are East, North, and
    /// Up (ENU), in row-major order.
    ///
    /// Beware: this coordinate system exhibits singularities at the poles.
    pub position_covariance: [f64; 9],

    // This member is not documented.
    #[allow(missing_docs)]
    pub position_covariance_type: u8,
}

impl NavSatFix {
    /// If the covariance of the fix is known, fill it in completely. If the
    /// GPS receiver provides the variance of each measurement, put them
    /// along the diagonal. If only Dilution of Precision is available,
    /// estimate an approximate covariance from that.
    pub const COVARIANCE_TYPE_UNKNOWN: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const COVARIANCE_TYPE_APPROXIMATED: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const COVARIANCE_TYPE_DIAGONAL_KNOWN: u8 = 2;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const COVARIANCE_TYPE_KNOWN: u8 = 3;
}

impl Default for NavSatFix {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::NavSatFix::default())
    }
}

impl rosidl_runtime_rs::Message for NavSatFix {
    type RmwMsg = super::msg::rmw::NavSatFix;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                status: super::msg::NavSatStatus::into_rmw_message(std::borrow::Cow::Owned(
                    msg.status,
                ))
                .into_owned(),
                latitude: msg.latitude,
                longitude: msg.longitude,
                altitude: msg.altitude,
                position_covariance: msg.position_covariance,
                position_covariance_type: msg.position_covariance_type,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                status: super::msg::NavSatStatus::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.status,
                ))
                .into_owned(),
                latitude: msg.latitude,
                longitude: msg.longitude,
                altitude: msg.altitude,
                position_covariance: msg.position_covariance,
                position_covariance_type: msg.position_covariance_type,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            status: super::msg::NavSatStatus::from_rmw_message(msg.status),
            latitude: msg.latitude,
            longitude: msg.longitude,
            altitude: msg.altitude,
            position_covariance: msg.position_covariance,
            position_covariance_type: msg.position_covariance_type,
        }
    }
}

// Corresponds to sensor_msgs__msg__NavSatStatus
/// Navigation Satellite fix status for any Global Navigation Satellite System.
///
/// Whether to output an augmented fix is determined by both the fix
/// type and the last time differential corrections were received.  A
/// fix is valid when status >= STATUS_FIX.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct NavSatStatus {
    // This member is not documented.
    #[allow(missing_docs)]
    pub status: i8,

    // This member is not documented.
    #[allow(missing_docs)]
    pub service: u16,
}

impl NavSatStatus {
    /// unable to fix position
    pub const STATUS_NO_FIX: i8 = -1;

    /// unaugmented fix
    pub const STATUS_FIX: i8 = 0;

    /// with satellite-based augmentation
    pub const STATUS_SBAS_FIX: i8 = 1;

    /// with ground-based augmentation
    pub const STATUS_GBAS_FIX: i8 = 2;

    /// Bits defining which Global Navigation Satellite System signals were
    /// used by the receiver.
    pub const SERVICE_GPS: u16 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const SERVICE_GLONASS: u16 = 2;

    /// includes BeiDou.
    pub const SERVICE_COMPASS: u16 = 4;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const SERVICE_GALILEO: u16 = 8;
}

impl Default for NavSatStatus {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::NavSatStatus::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for NavSatStatus {
    type RmwMsg = super::msg::rmw::NavSatStatus;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                status: msg.status,
                service: msg.service,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                status: msg.status,
                service: msg.service,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            status: msg.status,
            service: msg.service,
        }
    }
}

// Corresponds to sensor_msgs__msg__PointCloud
/// THIS MESSAGE IS DEPRECATED AS OF FOXY
/// Please use sensor_msgs/PointCloud2

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointCloud {
    /// This message holds a collection of 3d points, plus optional additional
    /// information about each point.
    /// Time of sensor data acquisition, coordinate frame ID.
    pub header: std_msgs::msg::Header,

    /// Array of 3d points. Each Point32 should be interpreted as a 3d point
    /// in the frame given in the header.
    pub points: Vec<geometry_msgs::msg::Point32>,

    /// Each channel should have the same number of elements as points array,
    /// and the data in each channel should correspond 1:1 with each point.
    /// Channel names in common practice are listed in ChannelFloat32.msg.
    pub channels: Vec<super::msg::ChannelFloat32>,
}

impl Default for PointCloud {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::PointCloud::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for PointCloud {
    type RmwMsg = super::msg::rmw::PointCloud;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                points: msg
                    .points
                    .into_iter()
                    .map(|elem| {
                        geometry_msgs::msg::Point32::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
                channels: msg
                    .channels
                    .into_iter()
                    .map(|elem| {
                        super::msg::ChannelFloat32::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                points: msg
                    .points
                    .iter()
                    .map(|elem| {
                        geometry_msgs::msg::Point32::into_rmw_message(std::borrow::Cow::Borrowed(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
                channels: msg
                    .channels
                    .iter()
                    .map(|elem| {
                        super::msg::ChannelFloat32::into_rmw_message(std::borrow::Cow::Borrowed(
                            elem,
                        ))
                        .into_owned()
                    })
                    .collect(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            points: msg
                .points
                .into_iter()
                .map(geometry_msgs::msg::Point32::from_rmw_message)
                .collect(),
            channels: msg
                .channels
                .into_iter()
                .map(super::msg::ChannelFloat32::from_rmw_message)
                .collect(),
        }
    }
}

// Corresponds to sensor_msgs__msg__PointCloud2
/// This message holds a collection of N-dimensional points, which may
/// contain additional information such as normals, intensity, etc. The
/// point data is stored as a binary blob, its layout described by the
/// contents of the "fields" array.
///
/// The point cloud data may be organized 2d (image-like) or 1d (unordered).
/// Point clouds organized as 2d images may be produced by camera depth sensors
/// such as stereo or time-of-flight.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointCloud2 {
    /// Time of sensor data acquisition, and the coordinate frame ID (for 3d points).
    pub header: std_msgs::msg::Header,

    /// 2D structure of the point cloud. If the cloud is unordered, height is
    /// 1 and width is the length of the point cloud.
    pub height: u32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub width: u32,

    /// Describes the channels and their layout in the binary data blob.
    pub fields: Vec<super::msg::PointField>,

    /// Is this data bigendian?
    pub is_bigendian: bool,

    /// Length of a point in bytes
    pub point_step: u32,

    /// Length of a row in bytes
    pub row_step: u32,

    /// Actual point data, size is (row_step*height)
    pub data: Vec<u8>,

    /// True if there are no invalid points
    pub is_dense: bool,
}

impl Default for PointCloud2 {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::PointCloud2::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for PointCloud2 {
    type RmwMsg = super::msg::rmw::PointCloud2;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                fields: msg
                    .fields
                    .into_iter()
                    .map(|elem| {
                        super::msg::PointField::into_rmw_message(std::borrow::Cow::Owned(elem))
                            .into_owned()
                    })
                    .collect(),
                is_bigendian: msg.is_bigendian,
                point_step: msg.point_step,
                row_step: msg.row_step,
                data: msg.data.into(),
                is_dense: msg.is_dense,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                height: msg.height,
                width: msg.width,
                fields: msg
                    .fields
                    .iter()
                    .map(|elem| {
                        super::msg::PointField::into_rmw_message(std::borrow::Cow::Borrowed(elem))
                            .into_owned()
                    })
                    .collect(),
                is_bigendian: msg.is_bigendian,
                point_step: msg.point_step,
                row_step: msg.row_step,
                data: msg.data.as_slice().into(),
                is_dense: msg.is_dense,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            height: msg.height,
            width: msg.width,
            fields: msg
                .fields
                .into_iter()
                .map(super::msg::PointField::from_rmw_message)
                .collect(),
            is_bigendian: msg.is_bigendian,
            point_step: msg.point_step,
            row_step: msg.row_step,
            data: msg.data.into_iter().collect(),
            is_dense: msg.is_dense,
        }
    }
}

// Corresponds to sensor_msgs__msg__PointField
/// This message holds the description of one point entry in the
/// PointCloud2 message format.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointField {
    /// Common PointField names are x, y, z, intensity, rgb, rgba
    /// Name of field
    pub name: std::string::String,

    /// Offset from start of point struct
    pub offset: u32,

    /// Datatype enumeration, see above
    pub datatype: u8,

    /// How many elements in the field
    pub count: u32,
}

impl PointField {
    // This constant is not documented.
    #[allow(missing_docs)]
    pub const INT8: u8 = 1;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const UINT8: u8 = 2;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const INT16: u8 = 3;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const UINT16: u8 = 4;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const INT32: u8 = 5;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const UINT32: u8 = 6;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const FLOAT32: u8 = 7;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const FLOAT64: u8 = 8;
}

impl Default for PointField {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::PointField::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for PointField {
    type RmwMsg = super::msg::rmw::PointField;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                name: msg.name.as_str().into(),
                offset: msg.offset,
                datatype: msg.datatype,
                count: msg.count,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                name: msg.name.as_str().into(),
                offset: msg.offset,
                datatype: msg.datatype,
                count: msg.count,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            name: msg.name.to_string(),
            offset: msg.offset,
            datatype: msg.datatype,
            count: msg.count,
        }
    }
}

// Corresponds to sensor_msgs__msg__Range
/// Single range reading from an active ranger that emits energy and reports
/// one range reading that is valid along an arc at the distance measured.
/// This message is  not appropriate for laser scanners. See the LaserScan
/// message if you are working with a laser scanner.
///
/// This message also can represent a fixed-distance (binary) ranger.  This
/// sensor will have min_range===max_range===distance of detection.
/// These sensors follow REP 117 and will output -Inf if the object is detected
/// and +Inf if the object is outside of the detection range.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Range {
    /// timestamp in the header is the time the ranger
    /// returned the distance reading
    pub header: std_msgs::msg::Header,

    /// the type of radiation used by the sensor
    /// (sound, IR, etc)
    pub radiation_type: u8,

    /// the size of the arc that the distance reading is
    /// valid for
    /// the object causing the range reading may have
    /// been anywhere within -field_of_view/2 and
    /// field_of_view/2 at the measured range.
    /// 0 angle corresponds to the x-axis of the sensor.
    pub field_of_view: f32,

    /// minimum range value
    pub min_range: f32,

    /// maximum range value
    /// Fixed distance rangers require min_range==max_range
    pub max_range: f32,

    /// range data
    /// (Note: values < range_min or > range_max should be discarded)
    /// Fixed distance rangers only output -Inf or +Inf.
    /// -Inf represents a detection within fixed distance.
    /// (Detection too close to the sensor to quantify)
    /// +Inf represents no detection within the fixed distance.
    /// (Object out of range)
    pub range: f32,
}

impl Range {
    /// Radiation type enums
    /// If you want a value added to this list, send an email to the ros-users list
    pub const ULTRASOUND: u8 = 0;

    // This constant is not documented.
    #[allow(missing_docs)]
    pub const INFRARED: u8 = 1;
}

impl Default for Range {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Range::default())
    }
}

impl rosidl_runtime_rs::Message for Range {
    type RmwMsg = super::msg::rmw::Range;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                radiation_type: msg.radiation_type,
                field_of_view: msg.field_of_view,
                min_range: msg.min_range,
                max_range: msg.max_range,
                range: msg.range,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                radiation_type: msg.radiation_type,
                field_of_view: msg.field_of_view,
                min_range: msg.min_range,
                max_range: msg.max_range,
                range: msg.range,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            radiation_type: msg.radiation_type,
            field_of_view: msg.field_of_view,
            min_range: msg.min_range,
            max_range: msg.max_range,
            range: msg.range,
        }
    }
}

// Corresponds to sensor_msgs__msg__RegionOfInterest
/// This message is used to specify a region of interest within an image.
///
/// When used to specify the ROI setting of the camera when the image was
/// taken, the height and width fields should either match the height and
/// width fields for the associated image; or height = width = 0
/// indicates that the full resolution image was captured.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RegionOfInterest {
    /// Leftmost pixel of the ROI
    /// (0 if the ROI includes the left edge of the image)
    pub x_offset: u32,

    /// Topmost pixel of the ROI
    /// (0 if the ROI includes the top edge of the image)
    pub y_offset: u32,

    /// Height of ROI
    pub height: u32,

    /// Width of ROI
    pub width: u32,

    /// True if a distinct rectified ROI should be calculated from the "raw"
    /// ROI in this message. Typically this should be False if the full image
    /// is captured (ROI not used), and True if a subwindow is captured (ROI
    /// used).
    pub do_rectify: bool,
}

impl Default for RegionOfInterest {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::RegionOfInterest::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for RegionOfInterest {
    type RmwMsg = super::msg::rmw::RegionOfInterest;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                x_offset: msg.x_offset,
                y_offset: msg.y_offset,
                height: msg.height,
                width: msg.width,
                do_rectify: msg.do_rectify,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                x_offset: msg.x_offset,
                y_offset: msg.y_offset,
                height: msg.height,
                width: msg.width,
                do_rectify: msg.do_rectify,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            x_offset: msg.x_offset,
            y_offset: msg.y_offset,
            height: msg.height,
            width: msg.width,
            do_rectify: msg.do_rectify,
        }
    }
}

// Corresponds to sensor_msgs__msg__RelativeHumidity
/// Single reading from a relative humidity sensor.
/// Defines the ratio of partial pressure of water vapor to the saturated vapor
/// pressure at a temperature.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RelativeHumidity {
    /// timestamp of the measurement
    /// frame_id is the location of the humidity sensor
    pub header: std_msgs::msg::Header,

    /// Expression of the relative humidity
    /// from 0.0 to 1.0.
    /// 0.0 is no partial pressure of water vapor
    /// 1.0 represents partial pressure of saturation
    pub relative_humidity: f64,

    /// 0 is interpreted as variance unknown
    pub variance: f64,
}

impl Default for RelativeHumidity {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::RelativeHumidity::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for RelativeHumidity {
    type RmwMsg = super::msg::rmw::RelativeHumidity;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                relative_humidity: msg.relative_humidity,
                variance: msg.variance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                relative_humidity: msg.relative_humidity,
                variance: msg.variance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            relative_humidity: msg.relative_humidity,
            variance: msg.variance,
        }
    }
}

// Corresponds to sensor_msgs__msg__Temperature
/// Single temperature reading.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Temperature {
    /// timestamp is the time the temperature was measured
    /// frame_id is the location of the temperature reading
    pub header: std_msgs::msg::Header,

    /// Measurement of the Temperature in Degrees Celsius.
    pub temperature: f64,

    /// 0 is interpreted as variance unknown.
    pub variance: f64,
}

impl Default for Temperature {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::Temperature::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for Temperature {
    type RmwMsg = super::msg::rmw::Temperature;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                temperature: msg.temperature,
                variance: msg.variance,
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                temperature: msg.temperature,
                variance: msg.variance,
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            temperature: msg.temperature,
            variance: msg.variance,
        }
    }
}

// Corresponds to sensor_msgs__msg__TimeReference
/// Measurement from an external time source not actively synchronized with the system clock.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TimeReference {
    /// stamp is system time for which measurement was valid
    /// frame_id is not used
    pub header: std_msgs::msg::Header,

    /// corresponding time from this external source
    pub time_ref: builtin_interfaces::msg::Time,

    /// (optional) name of time source
    pub source: std::string::String,
}

impl Default for TimeReference {
    fn default() -> Self {
        <Self as rosidl_runtime_rs::Message>::from_rmw_message(
            super::msg::rmw::TimeReference::default(),
        )
    }
}

impl rosidl_runtime_rs::Message for TimeReference {
    type RmwMsg = super::msg::rmw::TimeReference;

    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        match msg_cow {
            std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(
                    msg.header,
                ))
                .into_owned(),
                time_ref: builtin_interfaces::msg::Time::into_rmw_message(std::borrow::Cow::Owned(
                    msg.time_ref,
                ))
                .into_owned(),
                source: msg.source.as_str().into(),
            }),
            std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
                header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(
                    &msg.header,
                ))
                .into_owned(),
                time_ref: builtin_interfaces::msg::Time::into_rmw_message(
                    std::borrow::Cow::Borrowed(&msg.time_ref),
                )
                .into_owned(),
                source: msg.source.as_str().into(),
            }),
        }
    }

    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        Self {
            header: std_msgs::msg::Header::from_rmw_message(msg.header),
            time_ref: builtin_interfaces::msg::Time::from_rmw_message(msg.time_ref),
            source: msg.source.to_string(),
        }
    }
}
