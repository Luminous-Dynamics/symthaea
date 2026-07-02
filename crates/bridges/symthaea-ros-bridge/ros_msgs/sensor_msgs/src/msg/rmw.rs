#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__BatteryState(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__BatteryState__init(msg: *mut BatteryState) -> bool;
    fn sensor_msgs__msg__BatteryState__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<BatteryState>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__BatteryState__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<BatteryState>,
    );
    fn sensor_msgs__msg__BatteryState__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<BatteryState>,
        out_seq: *mut rosidl_runtime_rs::Sequence<BatteryState>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__BatteryState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
// This struct is not documented.
#[allow(missing_docs)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct BatteryState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

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
    pub cell_voltage: rosidl_runtime_rs::Sequence<f32>,

    /// An array of individual cell temperatures for each cell in the pack
    /// If individual temperatures unknown but number of cells known set each to NaN
    pub cell_temperature: rosidl_runtime_rs::Sequence<f32>,

    /// The location into which the battery is inserted. (slot number or plug)
    pub location: rosidl_runtime_rs::String,

    /// The best approximation of the battery serial number
    pub serial_number: rosidl_runtime_rs::String,
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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__BatteryState__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__BatteryState__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for BatteryState {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__BatteryState__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__BatteryState__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__BatteryState__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for BatteryState {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for BatteryState
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/BatteryState";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__BatteryState()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__CameraInfo(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__CameraInfo__init(msg: *mut CameraInfo) -> bool;
    fn sensor_msgs__msg__CameraInfo__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<CameraInfo>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__CameraInfo__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<CameraInfo>,
    );
    fn sensor_msgs__msg__CameraInfo__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<CameraInfo>,
        out_seq: *mut rosidl_runtime_rs::Sequence<CameraInfo>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__CameraInfo
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
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
    pub header: std_msgs::msg::rmw::Header,

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
    pub distortion_model: rosidl_runtime_rs::String,

    /// The distortion parameters, size depending on the distortion model.
    /// For "plumb_bob", the 5 parameters are: (k1, k2, t1, t2, k3).
    pub d: rosidl_runtime_rs::Sequence<f64>,

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
    pub roi: super::super::msg::rmw::RegionOfInterest,
}

impl Default for CameraInfo {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__CameraInfo__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__CameraInfo__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for CameraInfo {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CameraInfo__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CameraInfo__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CameraInfo__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for CameraInfo {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for CameraInfo
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/CameraInfo";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__CameraInfo()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__ChannelFloat32(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__ChannelFloat32__init(msg: *mut ChannelFloat32) -> bool;
    fn sensor_msgs__msg__ChannelFloat32__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<ChannelFloat32>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__ChannelFloat32__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<ChannelFloat32>,
    );
    fn sensor_msgs__msg__ChannelFloat32__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<ChannelFloat32>,
        out_seq: *mut rosidl_runtime_rs::Sequence<ChannelFloat32>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__ChannelFloat32
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct ChannelFloat32 {
    /// The channel name should give semantics of the channel (e.g.
    /// "intensity" instead of "value").
    pub name: rosidl_runtime_rs::String,

    /// The values array should be 1-1 with the elements of the associated
    /// PointCloud.
    pub values: rosidl_runtime_rs::Sequence<f32>,
}

impl Default for ChannelFloat32 {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__ChannelFloat32__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__ChannelFloat32__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for ChannelFloat32 {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__ChannelFloat32__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__ChannelFloat32__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__ChannelFloat32__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for ChannelFloat32 {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for ChannelFloat32
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/ChannelFloat32";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__ChannelFloat32(
            )
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__CompressedImage(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__CompressedImage__init(msg: *mut CompressedImage) -> bool;
    fn sensor_msgs__msg__CompressedImage__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<CompressedImage>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__CompressedImage__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<CompressedImage>,
    );
    fn sensor_msgs__msg__CompressedImage__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<CompressedImage>,
        out_seq: *mut rosidl_runtime_rs::Sequence<CompressedImage>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__CompressedImage
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message contains a compressed image.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct CompressedImage {
    /// Header timestamp should be acquisition time of image
    /// Header frame_id should be optical frame of camera
    /// origin of frame should be optical center of cameara
    /// +x should point to the right in the image
    /// +y should point down in the image
    /// +z should point into to plane of the image
    pub header: std_msgs::msg::rmw::Header,

    /// Specifies the format of the data
    ///   Acceptable values:
    ///     jpeg, png, tiff
    pub format: rosidl_runtime_rs::String,

    /// Compressed image buffer
    pub data: rosidl_runtime_rs::Sequence<u8>,
}

impl Default for CompressedImage {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__CompressedImage__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__CompressedImage__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for CompressedImage {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CompressedImage__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CompressedImage__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__CompressedImage__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for CompressedImage {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for CompressedImage
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/CompressedImage";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__CompressedImage(
            )
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__FluidPressure(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__FluidPressure__init(msg: *mut FluidPressure) -> bool;
    fn sensor_msgs__msg__FluidPressure__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<FluidPressure>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__FluidPressure__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<FluidPressure>,
    );
    fn sensor_msgs__msg__FluidPressure__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<FluidPressure>,
        out_seq: *mut rosidl_runtime_rs::Sequence<FluidPressure>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__FluidPressure
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single pressure reading.  This message is appropriate for measuring the
/// pressure inside of a fluid (air, water, etc).  This also includes
/// atmospheric or barometric pressure.
///
/// This message is not appropriate for force/pressure contact sensors.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct FluidPressure {
    /// timestamp of the measurement
    /// frame_id is the location of the pressure sensor
    pub header: std_msgs::msg::rmw::Header,

    /// Absolute pressure reading in Pascals.
    pub fluid_pressure: f64,

    /// 0 is interpreted as variance unknown
    pub variance: f64,
}

impl Default for FluidPressure {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__FluidPressure__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__FluidPressure__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for FluidPressure {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__FluidPressure__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__FluidPressure__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__FluidPressure__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for FluidPressure {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for FluidPressure
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/FluidPressure";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__FluidPressure()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Illuminance(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Illuminance__init(msg: *mut Illuminance) -> bool;
    fn sensor_msgs__msg__Illuminance__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Illuminance>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Illuminance__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Illuminance>,
    );
    fn sensor_msgs__msg__Illuminance__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Illuminance>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Illuminance>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Illuminance
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Illuminance {
    /// timestamp is the time the illuminance was measured
    /// frame_id is the location and direction of the reading
    pub header: std_msgs::msg::rmw::Header,

    /// Measurement of the Photometric Illuminance in Lux.
    pub illuminance: f64,

    /// 0 is interpreted as variance unknown
    pub variance: f64,
}

impl Default for Illuminance {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Illuminance__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Illuminance__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Illuminance {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Illuminance__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Illuminance__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Illuminance__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Illuminance {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Illuminance
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Illuminance";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Illuminance()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Image(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Image__init(msg: *mut Image) -> bool;
    fn sensor_msgs__msg__Image__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Image>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Image__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Image>);
    fn sensor_msgs__msg__Image__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Image>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Image>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Image
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message contains an uncompressed image
/// (0, 0) is at top-left corner of image

#[repr(C)]
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
    pub header: std_msgs::msg::rmw::Header,

    /// image height, that is, number of rows
    pub height: u32,

    /// image width, that is, number of columns
    pub width: u32,

    /// The legal values for encoding are in file src/image_encodings.cpp
    /// If you want to standardize a new string format, join
    /// ros-users@lists.ros.org and send an email proposing a new encoding.
    /// Encoding of pixels -- channel meaning, ordering, size
    /// taken from the list of strings in include/sensor_msgs/image_encodings.hpp
    pub encoding: rosidl_runtime_rs::String,

    /// is this data bigendian?
    pub is_bigendian: u8,

    /// Full row length in bytes
    pub step: u32,

    /// actual matrix data, size is (step * rows)
    pub data: rosidl_runtime_rs::Sequence<u8>,
}

impl Default for Image {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Image__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Image__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Image {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Image__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Image__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Image__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Image {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Image
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Image";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Image() }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Imu(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Imu__init(msg: *mut Imu) -> bool;
    fn sensor_msgs__msg__Imu__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Imu>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Imu__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Imu>);
    fn sensor_msgs__msg__Imu__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Imu>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Imu>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Imu
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Imu {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub orientation: geometry_msgs::msg::rmw::Quaternion,

    /// Row major about x, y, z axes
    pub orientation_covariance: [f64; 9],

    // This member is not documented.
    #[allow(missing_docs)]
    pub angular_velocity: geometry_msgs::msg::rmw::Vector3,

    /// Row major about x, y, z axes
    pub angular_velocity_covariance: [f64; 9],

    // This member is not documented.
    #[allow(missing_docs)]
    pub linear_acceleration: geometry_msgs::msg::rmw::Vector3,

    /// Row major x, y z
    pub linear_acceleration_covariance: [f64; 9],
}

impl Default for Imu {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Imu__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Imu__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Imu {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Imu__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Imu__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Imu__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Imu {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Imu
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Imu";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Imu() }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JointState(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__JointState__init(msg: *mut JointState) -> bool;
    fn sensor_msgs__msg__JointState__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<JointState>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__JointState__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<JointState>,
    );
    fn sensor_msgs__msg__JointState__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<JointState>,
        out_seq: *mut rosidl_runtime_rs::Sequence<JointState>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__JointState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub name: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub position: rosidl_runtime_rs::Sequence<f64>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: rosidl_runtime_rs::Sequence<f64>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub effort: rosidl_runtime_rs::Sequence<f64>,
}

impl Default for JointState {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__JointState__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__JointState__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for JointState {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JointState__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JointState__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JointState__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for JointState {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for JointState
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/JointState";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JointState()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Joy(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Joy__init(msg: *mut Joy) -> bool;
    fn sensor_msgs__msg__Joy__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Joy>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Joy__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Joy>);
    fn sensor_msgs__msg__Joy__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Joy>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Joy>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Joy
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Reports the state of a joystick's axes and buttons.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Joy {
    /// The timestamp is the time at which data is received from the joystick.
    pub header: std_msgs::msg::rmw::Header,

    /// The axes measurements from a joystick.
    pub axes: rosidl_runtime_rs::Sequence<f32>,

    /// The buttons measurements from a joystick.
    pub buttons: rosidl_runtime_rs::Sequence<i32>,
}

impl Default for Joy {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Joy__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Joy__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Joy {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Joy__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Joy__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Joy__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Joy {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Joy
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Joy";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Joy() }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JoyFeedback(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__JoyFeedback__init(msg: *mut JoyFeedback) -> bool;
    fn sensor_msgs__msg__JoyFeedback__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<JoyFeedback>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__JoyFeedback__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<JoyFeedback>,
    );
    fn sensor_msgs__msg__JoyFeedback__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<JoyFeedback>,
        out_seq: *mut rosidl_runtime_rs::Sequence<JoyFeedback>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__JoyFeedback
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Declare of the type of feedback

#[repr(C)]
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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__JoyFeedback__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__JoyFeedback__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for JoyFeedback {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedback__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedback__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedback__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for JoyFeedback {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for JoyFeedback
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/JoyFeedback";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JoyFeedback()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JoyFeedbackArray(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__JoyFeedbackArray__init(msg: *mut JoyFeedbackArray) -> bool;
    fn sensor_msgs__msg__JoyFeedbackArray__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<JoyFeedbackArray>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__JoyFeedbackArray__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<JoyFeedbackArray>,
    );
    fn sensor_msgs__msg__JoyFeedbackArray__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<JoyFeedbackArray>,
        out_seq: *mut rosidl_runtime_rs::Sequence<JoyFeedbackArray>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__JoyFeedbackArray
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message publishes values for multiple feedback at once.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JoyFeedbackArray {
    // This member is not documented.
    #[allow(missing_docs)]
    pub array: rosidl_runtime_rs::Sequence<super::super::msg::rmw::JoyFeedback>,
}

impl Default for JoyFeedbackArray {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__JoyFeedbackArray__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__JoyFeedbackArray__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for JoyFeedbackArray {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedbackArray__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedbackArray__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__JoyFeedbackArray__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for JoyFeedbackArray {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for JoyFeedbackArray
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/JoyFeedbackArray";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__JoyFeedbackArray()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__LaserEcho(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__LaserEcho__init(msg: *mut LaserEcho) -> bool;
    fn sensor_msgs__msg__LaserEcho__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<LaserEcho>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__LaserEcho__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<LaserEcho>,
    );
    fn sensor_msgs__msg__LaserEcho__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<LaserEcho>,
        out_seq: *mut rosidl_runtime_rs::Sequence<LaserEcho>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__LaserEcho
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message is a submessage of MultiEchoLaserScan and is not intended
/// to be used separately.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LaserEcho {
    /// Multiple values of ranges or intensities.
    /// Each array represents data from the same angle increment.
    pub echoes: rosidl_runtime_rs::Sequence<f32>,
}

impl Default for LaserEcho {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__LaserEcho__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__LaserEcho__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for LaserEcho {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserEcho__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserEcho__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserEcho__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for LaserEcho {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for LaserEcho
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/LaserEcho";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__LaserEcho()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__LaserScan(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__LaserScan__init(msg: *mut LaserScan) -> bool;
    fn sensor_msgs__msg__LaserScan__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<LaserScan>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__LaserScan__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<LaserScan>,
    );
    fn sensor_msgs__msg__LaserScan__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<LaserScan>,
        out_seq: *mut rosidl_runtime_rs::Sequence<LaserScan>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__LaserScan
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single scan from a planar laser range-finder
///
/// If you have another ranging device with different behavior (e.g. a sonar
/// array), please find or create a different message, since applications
/// will make fairly laser-specific assumptions about this data

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct LaserScan {
    /// timestamp in the header is the acquisition time of
    /// the first ray in the scan.
    ///
    /// in frame frame_id, angles are measured around
    /// the positive Z axis (counterclockwise, if Z is up)
    /// with zero angle being forward along the x axis
    pub header: std_msgs::msg::rmw::Header,

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
    pub ranges: rosidl_runtime_rs::Sequence<f32>,

    /// intensity data.  If your
    /// device does not provide intensities, please leave
    /// the array empty.
    pub intensities: rosidl_runtime_rs::Sequence<f32>,
}

impl Default for LaserScan {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__LaserScan__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__LaserScan__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for LaserScan {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserScan__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserScan__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__LaserScan__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for LaserScan {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for LaserScan
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/LaserScan";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__LaserScan()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MagneticField(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__MagneticField__init(msg: *mut MagneticField) -> bool;
    fn sensor_msgs__msg__MagneticField__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<MagneticField>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__MagneticField__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<MagneticField>,
    );
    fn sensor_msgs__msg__MagneticField__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<MagneticField>,
        out_seq: *mut rosidl_runtime_rs::Sequence<MagneticField>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__MagneticField
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Measurement of the Magnetic Field vector at a specific location.
///
/// If the covariance of the measurement is known, it should be filled in.
/// If all you know is the variance of each measurement, e.g. from the datasheet,
/// just put those along the diagonal.
/// A covariance matrix of all zeros will be interpreted as "covariance unknown",
/// and to use the data a covariance will have to be assumed or gotten from some
/// other source.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MagneticField {
    /// timestamp is the time the
    /// field was measured
    /// frame_id is the location and orientation
    /// of the field measurement
    pub header: std_msgs::msg::rmw::Header,

    /// x, y, and z components of the
    /// field vector in Tesla
    /// If your sensor does not output 3 axes,
    /// put NaNs in the components not reported.
    pub magnetic_field: geometry_msgs::msg::rmw::Vector3,

    /// Row major about x, y, z axes
    /// 0 is interpreted as variance unknown
    pub magnetic_field_covariance: [f64; 9],
}

impl Default for MagneticField {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__MagneticField__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__MagneticField__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for MagneticField {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MagneticField__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MagneticField__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MagneticField__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for MagneticField {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for MagneticField
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/MagneticField";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MagneticField()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MultiDOFJointState(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__MultiDOFJointState__init(msg: *mut MultiDOFJointState) -> bool;
    fn sensor_msgs__msg__MultiDOFJointState__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointState>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__MultiDOFJointState__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointState>,
    );
    fn sensor_msgs__msg__MultiDOFJointState__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<MultiDOFJointState>,
        out_seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointState>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__MultiDOFJointState
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

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

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointState {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub joint_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub transforms: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Transform>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Twist>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub wrench: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Wrench>,
}

impl Default for MultiDOFJointState {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__MultiDOFJointState__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__MultiDOFJointState__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for MultiDOFJointState {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiDOFJointState__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiDOFJointState__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiDOFJointState__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for MultiDOFJointState {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for MultiDOFJointState
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/MultiDOFJointState";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MultiDOFJointState()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MultiEchoLaserScan(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__MultiEchoLaserScan__init(msg: *mut MultiEchoLaserScan) -> bool;
    fn sensor_msgs__msg__MultiEchoLaserScan__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<MultiEchoLaserScan>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__MultiEchoLaserScan__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<MultiEchoLaserScan>,
    );
    fn sensor_msgs__msg__MultiEchoLaserScan__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<MultiEchoLaserScan>,
        out_seq: *mut rosidl_runtime_rs::Sequence<MultiEchoLaserScan>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__MultiEchoLaserScan
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single scan from a multi-echo planar laser range-finder
///
/// If you have another ranging device with different behavior (e.g. a sonar
/// array), please find or create a different message, since applications
/// will make fairly laser-specific assumptions about this data

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiEchoLaserScan {
    /// timestamp in the header is the acquisition time of
    /// the first ray in the scan.
    ///
    /// in frame frame_id, angles are measured around
    /// the positive Z axis (counterclockwise, if Z is up)
    /// with zero angle being forward along the x axis
    pub header: std_msgs::msg::rmw::Header,

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
    pub ranges: rosidl_runtime_rs::Sequence<super::super::msg::rmw::LaserEcho>,

    /// intensity data.  If your
    /// device does not provide intensities, please leave
    /// the array empty.
    pub intensities: rosidl_runtime_rs::Sequence<super::super::msg::rmw::LaserEcho>,
}

impl Default for MultiEchoLaserScan {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__MultiEchoLaserScan__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__MultiEchoLaserScan__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for MultiEchoLaserScan {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiEchoLaserScan__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiEchoLaserScan__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__MultiEchoLaserScan__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for MultiEchoLaserScan {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for MultiEchoLaserScan
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/MultiEchoLaserScan";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__MultiEchoLaserScan()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__NavSatFix(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__NavSatFix__init(msg: *mut NavSatFix) -> bool;
    fn sensor_msgs__msg__NavSatFix__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<NavSatFix>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__NavSatFix__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<NavSatFix>,
    );
    fn sensor_msgs__msg__NavSatFix__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<NavSatFix>,
        out_seq: *mut rosidl_runtime_rs::Sequence<NavSatFix>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__NavSatFix
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Navigation Satellite fix for any Global Navigation Satellite System
///
/// Specified using the WGS 84 reference ellipsoid

#[repr(C)]
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
    pub header: std_msgs::msg::rmw::Header,

    /// Satellite fix status information.
    pub status: super::super::msg::rmw::NavSatStatus,

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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__NavSatFix__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__NavSatFix__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for NavSatFix {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatFix__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatFix__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatFix__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for NavSatFix {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for NavSatFix
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/NavSatFix";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__NavSatFix()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__NavSatStatus(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__NavSatStatus__init(msg: *mut NavSatStatus) -> bool;
    fn sensor_msgs__msg__NavSatStatus__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<NavSatStatus>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__NavSatStatus__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<NavSatStatus>,
    );
    fn sensor_msgs__msg__NavSatStatus__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<NavSatStatus>,
        out_seq: *mut rosidl_runtime_rs::Sequence<NavSatStatus>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__NavSatStatus
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Navigation Satellite fix status for any Global Navigation Satellite System.
///
/// Whether to output an augmented fix is determined by both the fix
/// type and the last time differential corrections were received.  A
/// fix is valid when status >= STATUS_FIX.

#[repr(C)]
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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__NavSatStatus__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__NavSatStatus__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for NavSatStatus {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatStatus__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatStatus__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__NavSatStatus__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for NavSatStatus {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for NavSatStatus
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/NavSatStatus";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__NavSatStatus()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointCloud(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__PointCloud__init(msg: *mut PointCloud) -> bool;
    fn sensor_msgs__msg__PointCloud__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PointCloud>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__PointCloud__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PointCloud>,
    );
    fn sensor_msgs__msg__PointCloud__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PointCloud>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PointCloud>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__PointCloud
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// THIS MESSAGE IS DEPRECATED AS OF FOXY
/// Please use sensor_msgs/PointCloud2

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointCloud {
    /// This message holds a collection of 3d points, plus optional additional
    /// information about each point.
    /// Time of sensor data acquisition, coordinate frame ID.
    pub header: std_msgs::msg::rmw::Header,

    /// Array of 3d points. Each Point32 should be interpreted as a 3d point
    /// in the frame given in the header.
    pub points: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Point32>,

    /// Each channel should have the same number of elements as points array,
    /// and the data in each channel should correspond 1:1 with each point.
    /// Channel names in common practice are listed in ChannelFloat32.msg.
    pub channels: rosidl_runtime_rs::Sequence<super::super::msg::rmw::ChannelFloat32>,
}

impl Default for PointCloud {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__PointCloud__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__PointCloud__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PointCloud {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PointCloud {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PointCloud
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/PointCloud";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointCloud()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointCloud2(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__PointCloud2__init(msg: *mut PointCloud2) -> bool;
    fn sensor_msgs__msg__PointCloud2__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PointCloud2>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__PointCloud2__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PointCloud2>,
    );
    fn sensor_msgs__msg__PointCloud2__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PointCloud2>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PointCloud2>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__PointCloud2
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message holds a collection of N-dimensional points, which may
/// contain additional information such as normals, intensity, etc. The
/// point data is stored as a binary blob, its layout described by the
/// contents of the "fields" array.
///
/// The point cloud data may be organized 2d (image-like) or 1d (unordered).
/// Point clouds organized as 2d images may be produced by camera depth sensors
/// such as stereo or time-of-flight.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointCloud2 {
    /// Time of sensor data acquisition, and the coordinate frame ID (for 3d points).
    pub header: std_msgs::msg::rmw::Header,

    /// 2D structure of the point cloud. If the cloud is unordered, height is
    /// 1 and width is the length of the point cloud.
    pub height: u32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub width: u32,

    /// Describes the channels and their layout in the binary data blob.
    pub fields: rosidl_runtime_rs::Sequence<super::super::msg::rmw::PointField>,

    /// Is this data bigendian?
    pub is_bigendian: bool,

    /// Length of a point in bytes
    pub point_step: u32,

    /// Length of a row in bytes
    pub row_step: u32,

    /// Actual point data, size is (row_step*height)
    pub data: rosidl_runtime_rs::Sequence<u8>,

    /// True if there are no invalid points
    pub is_dense: bool,
}

impl Default for PointCloud2 {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__PointCloud2__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__PointCloud2__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PointCloud2 {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud2__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud2__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointCloud2__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PointCloud2 {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PointCloud2
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/PointCloud2";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointCloud2()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointField(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__PointField__init(msg: *mut PointField) -> bool;
    fn sensor_msgs__msg__PointField__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PointField>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__PointField__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PointField>,
    );
    fn sensor_msgs__msg__PointField__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PointField>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PointField>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__PointField
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message holds the description of one point entry in the
/// PointCloud2 message format.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointField {
    /// Common PointField names are x, y, z, intensity, rgb, rgba
    /// Name of field
    pub name: rosidl_runtime_rs::String,

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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__PointField__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__PointField__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PointField {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointField__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointField__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__PointField__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PointField {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PointField
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/PointField";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__PointField()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Range(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Range__init(msg: *mut Range) -> bool;
    fn sensor_msgs__msg__Range__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Range>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Range__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Range>);
    fn sensor_msgs__msg__Range__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Range>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Range>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Range
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single range reading from an active ranger that emits energy and reports
/// one range reading that is valid along an arc at the distance measured.
/// This message is  not appropriate for laser scanners. See the LaserScan
/// message if you are working with a laser scanner.
///
/// This message also can represent a fixed-distance (binary) ranger.  This
/// sensor will have min_range===max_range===distance of detection.
/// These sensors follow REP 117 and will output -Inf if the object is detected
/// and +Inf if the object is outside of the detection range.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Range {
    /// timestamp in the header is the time the ranger
    /// returned the distance reading
    pub header: std_msgs::msg::rmw::Header,

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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Range__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Range__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Range {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Range__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Range__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Range__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Range {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Range
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Range";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Range() }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__RegionOfInterest(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__RegionOfInterest__init(msg: *mut RegionOfInterest) -> bool;
    fn sensor_msgs__msg__RegionOfInterest__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<RegionOfInterest>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__RegionOfInterest__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<RegionOfInterest>,
    );
    fn sensor_msgs__msg__RegionOfInterest__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<RegionOfInterest>,
        out_seq: *mut rosidl_runtime_rs::Sequence<RegionOfInterest>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__RegionOfInterest
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message is used to specify a region of interest within an image.
///
/// When used to specify the ROI setting of the camera when the image was
/// taken, the height and width fields should either match the height and
/// width fields for the associated image; or height = width = 0
/// indicates that the full resolution image was captured.

#[repr(C)]
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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__RegionOfInterest__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__RegionOfInterest__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for RegionOfInterest {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RegionOfInterest__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RegionOfInterest__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RegionOfInterest__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for RegionOfInterest {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for RegionOfInterest
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/RegionOfInterest";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__RegionOfInterest()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__RelativeHumidity(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__RelativeHumidity__init(msg: *mut RelativeHumidity) -> bool;
    fn sensor_msgs__msg__RelativeHumidity__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<RelativeHumidity>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__RelativeHumidity__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<RelativeHumidity>,
    );
    fn sensor_msgs__msg__RelativeHumidity__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<RelativeHumidity>,
        out_seq: *mut rosidl_runtime_rs::Sequence<RelativeHumidity>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__RelativeHumidity
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single reading from a relative humidity sensor.
/// Defines the ratio of partial pressure of water vapor to the saturated vapor
/// pressure at a temperature.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct RelativeHumidity {
    /// timestamp of the measurement
    /// frame_id is the location of the humidity sensor
    pub header: std_msgs::msg::rmw::Header,

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
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__RelativeHumidity__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__RelativeHumidity__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for RelativeHumidity {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RelativeHumidity__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RelativeHumidity__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__RelativeHumidity__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for RelativeHumidity {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for RelativeHumidity
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/RelativeHumidity";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__RelativeHumidity()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Temperature(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__Temperature__init(msg: *mut Temperature) -> bool;
    fn sensor_msgs__msg__Temperature__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Temperature>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__Temperature__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Temperature>,
    );
    fn sensor_msgs__msg__Temperature__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Temperature>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Temperature>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__Temperature
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Single temperature reading.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Temperature {
    /// timestamp is the time the temperature was measured
    /// frame_id is the location of the temperature reading
    pub header: std_msgs::msg::rmw::Header,

    /// Measurement of the Temperature in Degrees Celsius.
    pub temperature: f64,

    /// 0 is interpreted as variance unknown.
    pub variance: f64,
}

impl Default for Temperature {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__Temperature__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__Temperature__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Temperature {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Temperature__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Temperature__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__Temperature__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Temperature {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Temperature
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/Temperature";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__Temperature()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__TimeReference(
    ) -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__msg__TimeReference__init(msg: *mut TimeReference) -> bool;
    fn sensor_msgs__msg__TimeReference__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<TimeReference>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__msg__TimeReference__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<TimeReference>,
    );
    fn sensor_msgs__msg__TimeReference__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<TimeReference>,
        out_seq: *mut rosidl_runtime_rs::Sequence<TimeReference>,
    ) -> bool;
}

// Corresponds to sensor_msgs__msg__TimeReference
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Measurement from an external time source not actively synchronized with the system clock.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TimeReference {
    /// stamp is system time for which measurement was valid
    /// frame_id is not used
    pub header: std_msgs::msg::rmw::Header,

    /// corresponding time from this external source
    pub time_ref: builtin_interfaces::msg::rmw::Time,

    /// (optional) name of time source
    pub source: rosidl_runtime_rs::String,
}

impl Default for TimeReference {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__msg__TimeReference__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__msg__TimeReference__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for TimeReference {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__TimeReference__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__TimeReference__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__msg__TimeReference__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for TimeReference {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for TimeReference
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/msg/TimeReference";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__msg__TimeReference()
        }
    }
}
