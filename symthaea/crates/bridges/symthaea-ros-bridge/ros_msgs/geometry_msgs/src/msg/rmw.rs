#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Accel()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Accel__init(msg: *mut Accel) -> bool;
    fn geometry_msgs__msg__Accel__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Accel>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Accel__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Accel>);
    fn geometry_msgs__msg__Accel__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Accel>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Accel>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Accel
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses acceleration in free space broken into its linear and angular parts.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Accel {
    // This member is not documented.
    #[allow(missing_docs)]
    pub linear: super::super::msg::rmw::Vector3,

    // This member is not documented.
    #[allow(missing_docs)]
    pub angular: super::super::msg::rmw::Vector3,
}

impl Default for Accel {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Accel__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Accel__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Accel {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Accel__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Accel__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Accel__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Accel {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Accel
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Accel";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Accel()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__AccelStamped__init(msg: *mut AccelStamped) -> bool;
    fn geometry_msgs__msg__AccelStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<AccelStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__AccelStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<AccelStamped>,
    );
    fn geometry_msgs__msg__AccelStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<AccelStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<AccelStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__AccelStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// An accel with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::super::msg::rmw::Accel,
}

impl Default for AccelStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__AccelStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__AccelStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for AccelStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for AccelStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for AccelStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/AccelStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelStamped(
            )
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelWithCovariance()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__AccelWithCovariance__init(msg: *mut AccelWithCovariance) -> bool;
    fn geometry_msgs__msg__AccelWithCovariance__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovariance>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__AccelWithCovariance__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovariance>,
    );
    fn geometry_msgs__msg__AccelWithCovariance__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<AccelWithCovariance>,
        out_seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovariance>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__AccelWithCovariance
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses acceleration in free space with uncertainty.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelWithCovariance {
    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::super::msg::rmw::Accel,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],
}

impl Default for AccelWithCovariance {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__AccelWithCovariance__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__AccelWithCovariance__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for AccelWithCovariance {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelWithCovariance__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelWithCovariance__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__AccelWithCovariance__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for AccelWithCovariance {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for AccelWithCovariance
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/AccelWithCovariance";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelWithCovariance()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelWithCovarianceStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__AccelWithCovarianceStamped__init(
        msg: *mut AccelWithCovarianceStamped,
    ) -> bool;
    fn geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovarianceStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovarianceStamped>,
    );
    fn geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<AccelWithCovarianceStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<AccelWithCovarianceStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__AccelWithCovarianceStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents an estimated accel with reference coordinate frame and timestamp.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelWithCovarianceStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::super::msg::rmw::AccelWithCovariance,
}

impl Default for AccelWithCovarianceStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__AccelWithCovarianceStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__AccelWithCovarianceStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for AccelWithCovarianceStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__init(seq as *mut _, size)
        }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__AccelWithCovarianceStamped__Sequence__copy(
                in_seq,
                out_seq as *mut _,
            )
        }
    }
}

impl rosidl_runtime_rs::Message for AccelWithCovarianceStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for AccelWithCovarianceStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/AccelWithCovarianceStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__AccelWithCovarianceStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Inertia()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Inertia__init(msg: *mut Inertia) -> bool;
    fn geometry_msgs__msg__Inertia__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Inertia>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Inertia__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Inertia>);
    fn geometry_msgs__msg__Inertia__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Inertia>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Inertia>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Inertia
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Mass

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Inertia {
    // This member is not documented.
    #[allow(missing_docs)]
    pub m: f64,

    /// Center of mass
    pub com: super::super::msg::rmw::Vector3,

    /// Inertia Tensor about the center of mass
    ///     | ixx ixy ixz |
    /// I = | ixy iyy iyz |
    ///     | ixz iyz izz |
    pub ixx: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub ixy: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub ixz: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub iyy: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub iyz: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub izz: f64,
}

impl Default for Inertia {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Inertia__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Inertia__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Inertia {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Inertia__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Inertia__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Inertia__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Inertia {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Inertia
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Inertia";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Inertia()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__InertiaStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__InertiaStamped__init(msg: *mut InertiaStamped) -> bool;
    fn geometry_msgs__msg__InertiaStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<InertiaStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__InertiaStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<InertiaStamped>,
    );
    fn geometry_msgs__msg__InertiaStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<InertiaStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<InertiaStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__InertiaStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// An Inertia with a time stamp and reference frame.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct InertiaStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub inertia: super::super::msg::rmw::Inertia,
}

impl Default for InertiaStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__InertiaStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__InertiaStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for InertiaStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__InertiaStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__InertiaStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__InertiaStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for InertiaStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for InertiaStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/InertiaStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__InertiaStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Point()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Point__init(msg: *mut Point) -> bool;
    fn geometry_msgs__msg__Point__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Point>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Point__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Point>);
    fn geometry_msgs__msg__Point__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Point>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Point>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Point
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This contains the position of a point in free space

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Point {
    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub z: f64,
}

impl Default for Point {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Point__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Point__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Point {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Point {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Point
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Point";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Point()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Point32()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Point32__init(msg: *mut Point32) -> bool;
    fn geometry_msgs__msg__Point32__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Point32>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Point32__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Point32>);
    fn geometry_msgs__msg__Point32__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Point32>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Point32>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Point32
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This contains the position of a point in free space(with 32 bits of precision).
/// It is recommended to use Point wherever possible instead of Point32.
///
/// This recommendation is to promote interoperability.
///
/// This message is designed to take up less space when sending
/// lots of points at once, as in the case of a PointCloud.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Point32 {
    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f32,

    // This member is not documented.
    #[allow(missing_docs)]
    pub z: f32,
}

impl Default for Point32 {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Point32__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Point32__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Point32 {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point32__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point32__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Point32__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Point32 {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Point32
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Point32";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Point32()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PointStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PointStamped__init(msg: *mut PointStamped) -> bool;
    fn geometry_msgs__msg__PointStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PointStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PointStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PointStamped>,
    );
    fn geometry_msgs__msg__PointStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PointStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PointStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PointStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a Point with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub point: super::super::msg::rmw::Point,
}

impl Default for PointStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PointStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PointStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PointStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PointStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PointStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PointStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PointStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PointStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PointStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PointStamped(
            )
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Polygon()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Polygon__init(msg: *mut Polygon) -> bool;
    fn geometry_msgs__msg__Polygon__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Polygon>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Polygon__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Polygon>);
    fn geometry_msgs__msg__Polygon__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Polygon>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Polygon>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Polygon
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A specification of a polygon where the first and last points are assumed to be connected

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Polygon {
    // This member is not documented.
    #[allow(missing_docs)]
    pub points: rosidl_runtime_rs::Sequence<super::super::msg::rmw::Point32>,
}

impl Default for Polygon {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Polygon__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Polygon__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Polygon {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Polygon__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Polygon__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Polygon__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Polygon {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Polygon
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Polygon";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Polygon()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonInstance()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PolygonInstance__init(msg: *mut PolygonInstance) -> bool;
    fn geometry_msgs__msg__PolygonInstance__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonInstance>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PolygonInstance__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonInstance>,
    );
    fn geometry_msgs__msg__PolygonInstance__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PolygonInstance>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PolygonInstance>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PolygonInstance
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A specification of a polygon where the first and last points are assumed to be connected
/// It includes a unique identification field for disambiguating multiple instances

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonInstance {
    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::super::msg::rmw::Polygon,

    // This member is not documented.
    #[allow(missing_docs)]
    pub id: i64,
}

impl Default for PolygonInstance {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PolygonInstance__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PolygonInstance__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PolygonInstance {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonInstance__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonInstance__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonInstance__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PolygonInstance {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PolygonInstance
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PolygonInstance";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonInstance()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonInstanceStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PolygonInstanceStamped__init(msg: *mut PolygonInstanceStamped) -> bool;
    fn geometry_msgs__msg__PolygonInstanceStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonInstanceStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PolygonInstanceStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonInstanceStamped>,
    );
    fn geometry_msgs__msg__PolygonInstanceStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PolygonInstanceStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PolygonInstanceStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PolygonInstanceStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a Polygon with reference coordinate frame and timestamp
/// It includes a unique identification field for disambiguating multiple instances

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonInstanceStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::super::msg::rmw::PolygonInstance,
}

impl Default for PolygonInstanceStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PolygonInstanceStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PolygonInstanceStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PolygonInstanceStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonInstanceStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonInstanceStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__PolygonInstanceStamped__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for PolygonInstanceStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PolygonInstanceStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PolygonInstanceStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonInstanceStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PolygonStamped__init(msg: *mut PolygonStamped) -> bool;
    fn geometry_msgs__msg__PolygonStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PolygonStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PolygonStamped>,
    );
    fn geometry_msgs__msg__PolygonStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PolygonStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PolygonStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PolygonStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a Polygon with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::super::msg::rmw::Polygon,
}

impl Default for PolygonStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PolygonStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PolygonStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PolygonStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PolygonStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PolygonStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PolygonStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PolygonStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PolygonStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Pose()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Pose__init(msg: *mut Pose) -> bool;
    fn geometry_msgs__msg__Pose__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Pose>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Pose__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Pose>);
    fn geometry_msgs__msg__Pose__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Pose>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Pose>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Pose
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A representation of pose in free space, composed of position and orientation.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose {
    // This member is not documented.
    #[allow(missing_docs)]
    pub position: super::super::msg::rmw::Point,

    // This member is not documented.
    #[allow(missing_docs)]
    pub orientation: super::super::msg::rmw::Quaternion,
}

impl Default for Pose {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Pose__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Pose__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Pose {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Pose {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Pose
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Pose";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Pose() }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Pose2D()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Pose2D__init(msg: *mut Pose2D) -> bool;
    fn geometry_msgs__msg__Pose2D__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Pose2D>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Pose2D__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Pose2D>);
    fn geometry_msgs__msg__Pose2D__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Pose2D>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Pose2D>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Pose2D
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Deprecated as of Foxy and will potentially be removed in any following release.
/// Please use the full 3D pose.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose2D {
    /// In general our recommendation is to use a full 3D representation of everything and for 2D specific applications make the appropriate projections into the plane for their calculations but optimally will preserve the 3D information during processing.
    /// If we have parallel copies of 2D datatypes every UI and other pipeline will end up needing to have dual interfaces to plot everything. And you will end up with not being able to use 3D tools for 2D use cases even if they're completely valid, as you'd have to reimplement it with different inputs and outputs. It's not particularly hard to plot the 2D pose or compute the yaw error for the Pose message and there are already tools and libraries that can do this for you.# This expresses a position and orientation on a 2D manifold.
    pub x: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub theta: f64,
}

impl Default for Pose2D {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Pose2D__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Pose2D__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Pose2D {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose2D__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose2D__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Pose2D__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Pose2D {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Pose2D
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Pose2D";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Pose2D()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseArray()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PoseArray__init(msg: *mut PoseArray) -> bool;
    fn geometry_msgs__msg__PoseArray__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PoseArray>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PoseArray__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PoseArray>,
    );
    fn geometry_msgs__msg__PoseArray__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PoseArray>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PoseArray>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PoseArray
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// An array of poses with a header for global reference.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseArray {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub poses: rosidl_runtime_rs::Sequence<super::super::msg::rmw::Pose>,
}

impl Default for PoseArray {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PoseArray__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PoseArray__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PoseArray {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseArray__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseArray__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseArray__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PoseArray {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PoseArray
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PoseArray";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseArray()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PoseStamped__init(msg: *mut PoseStamped) -> bool;
    fn geometry_msgs__msg__PoseStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PoseStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PoseStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PoseStamped>,
    );
    fn geometry_msgs__msg__PoseStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PoseStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PoseStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PoseStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A Pose with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::super::msg::rmw::Pose,
}

impl Default for PoseStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PoseStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PoseStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PoseStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PoseStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PoseStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PoseStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseWithCovariance()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PoseWithCovariance__init(msg: *mut PoseWithCovariance) -> bool;
    fn geometry_msgs__msg__PoseWithCovariance__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovariance>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PoseWithCovariance__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovariance>,
    );
    fn geometry_msgs__msg__PoseWithCovariance__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PoseWithCovariance>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovariance>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PoseWithCovariance
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a pose in free space with uncertainty.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseWithCovariance {
    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::super::msg::rmw::Pose,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],
}

impl Default for PoseWithCovariance {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PoseWithCovariance__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PoseWithCovariance__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PoseWithCovariance {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseWithCovariance__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseWithCovariance__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseWithCovariance__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for PoseWithCovariance {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PoseWithCovariance
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PoseWithCovariance";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseWithCovariance()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseWithCovarianceStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__PoseWithCovarianceStamped__init(
        msg: *mut PoseWithCovarianceStamped,
    ) -> bool;
    fn geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovarianceStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovarianceStamped>,
    );
    fn geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<PoseWithCovarianceStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<PoseWithCovarianceStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__PoseWithCovarianceStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses an estimated pose with a reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseWithCovarianceStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::super::msg::rmw::PoseWithCovariance,
}

impl Default for PoseWithCovarianceStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__PoseWithCovarianceStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__PoseWithCovarianceStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for PoseWithCovarianceStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__init(seq as *mut _, size)
        }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__PoseWithCovarianceStamped__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for PoseWithCovarianceStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for PoseWithCovarianceStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/PoseWithCovarianceStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__PoseWithCovarianceStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Quaternion()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Quaternion__init(msg: *mut Quaternion) -> bool;
    fn geometry_msgs__msg__Quaternion__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Quaternion>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Quaternion__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Quaternion>,
    );
    fn geometry_msgs__msg__Quaternion__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Quaternion>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Quaternion>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Quaternion
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents an orientation in free space in quaternion form.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Quaternion {
    // This member is not documented.
    #[allow(missing_docs)]
    pub x: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub z: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub w: f64,
}

impl Default for Quaternion {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Quaternion__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Quaternion__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Quaternion {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Quaternion__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Quaternion__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Quaternion__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Quaternion {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Quaternion
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Quaternion";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Quaternion()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__QuaternionStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__QuaternionStamped__init(msg: *mut QuaternionStamped) -> bool;
    fn geometry_msgs__msg__QuaternionStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<QuaternionStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__QuaternionStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<QuaternionStamped>,
    );
    fn geometry_msgs__msg__QuaternionStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<QuaternionStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<QuaternionStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__QuaternionStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents an orientation with reference coordinate frame and timestamp.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct QuaternionStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub quaternion: super::super::msg::rmw::Quaternion,
}

impl Default for QuaternionStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__QuaternionStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__QuaternionStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for QuaternionStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__QuaternionStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__QuaternionStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__QuaternionStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for QuaternionStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for QuaternionStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/QuaternionStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__QuaternionStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Transform()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Transform__init(msg: *mut Transform) -> bool;
    fn geometry_msgs__msg__Transform__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Transform>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Transform__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Transform>,
    );
    fn geometry_msgs__msg__Transform__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Transform>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Transform>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Transform
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents the transform between two coordinate frames in free space.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Transform {
    // This member is not documented.
    #[allow(missing_docs)]
    pub translation: super::super::msg::rmw::Vector3,

    // This member is not documented.
    #[allow(missing_docs)]
    pub rotation: super::super::msg::rmw::Quaternion,
}

impl Default for Transform {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Transform__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Transform__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Transform {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Transform__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Transform__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Transform__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Transform {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Transform
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Transform";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Transform()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TransformStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__TransformStamped__init(msg: *mut TransformStamped) -> bool;
    fn geometry_msgs__msg__TransformStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<TransformStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__TransformStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<TransformStamped>,
    );
    fn geometry_msgs__msg__TransformStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<TransformStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<TransformStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__TransformStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses a transform from coordinate frame header.frame_id
/// to the coordinate frame child_frame_id at the time of header.stamp
///
/// This message is mostly used by the
/// <a href="https://index.ros.org/p/tf2/">tf2</a> package.
/// See its documentation for more information.
///
/// The child_frame_id is necessary in addition to the frame_id
/// in the Header to communicate the full reference for the transform
/// in a self contained message.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TransformStamped {
    /// The frame id in the header is used as the reference frame of this transform.
    pub header: std_msgs::msg::rmw::Header,

    /// The frame id of the child frame to which this transform points.
    pub child_frame_id: rosidl_runtime_rs::String,

    /// Translation and rotation in 3-dimensions of child_frame_id from header.frame_id.
    pub transform: super::super::msg::rmw::Transform,
}

impl Default for TransformStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__TransformStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__TransformStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for TransformStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TransformStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TransformStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TransformStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for TransformStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for TransformStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/TransformStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TransformStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Twist()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Twist__init(msg: *mut Twist) -> bool;
    fn geometry_msgs__msg__Twist__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Twist>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Twist__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Twist>);
    fn geometry_msgs__msg__Twist__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Twist>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Twist>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Twist
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses velocity in free space broken into its linear and angular parts.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Twist {
    // This member is not documented.
    #[allow(missing_docs)]
    pub linear: super::super::msg::rmw::Vector3,

    // This member is not documented.
    #[allow(missing_docs)]
    pub angular: super::super::msg::rmw::Vector3,
}

impl Default for Twist {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Twist__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Twist__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Twist {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Twist__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Twist__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Twist__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Twist {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Twist
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Twist";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Twist()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__TwistStamped__init(msg: *mut TwistStamped) -> bool;
    fn geometry_msgs__msg__TwistStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<TwistStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__TwistStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<TwistStamped>,
    );
    fn geometry_msgs__msg__TwistStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<TwistStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<TwistStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__TwistStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A twist with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::super::msg::rmw::Twist,
}

impl Default for TwistStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__TwistStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__TwistStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for TwistStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for TwistStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for TwistStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/TwistStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistStamped(
            )
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistWithCovariance()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__TwistWithCovariance__init(msg: *mut TwistWithCovariance) -> bool;
    fn geometry_msgs__msg__TwistWithCovariance__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovariance>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__TwistWithCovariance__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovariance>,
    );
    fn geometry_msgs__msg__TwistWithCovariance__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<TwistWithCovariance>,
        out_seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovariance>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__TwistWithCovariance
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses velocity in free space with uncertainty.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistWithCovariance {
    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::super::msg::rmw::Twist,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],
}

impl Default for TwistWithCovariance {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__TwistWithCovariance__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__TwistWithCovariance__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for TwistWithCovariance {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistWithCovariance__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistWithCovariance__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__TwistWithCovariance__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for TwistWithCovariance {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for TwistWithCovariance
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/TwistWithCovariance";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistWithCovariance()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistWithCovarianceStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__TwistWithCovarianceStamped__init(
        msg: *mut TwistWithCovarianceStamped,
    ) -> bool;
    fn geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovarianceStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovarianceStamped>,
    );
    fn geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<TwistWithCovarianceStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<TwistWithCovarianceStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__TwistWithCovarianceStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents an estimated twist with reference coordinate frame and timestamp.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistWithCovarianceStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::super::msg::rmw::TwistWithCovariance,
}

impl Default for TwistWithCovarianceStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__TwistWithCovarianceStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__TwistWithCovarianceStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for TwistWithCovarianceStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__init(seq as *mut _, size)
        }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            geometry_msgs__msg__TwistWithCovarianceStamped__Sequence__copy(
                in_seq,
                out_seq as *mut _,
            )
        }
    }
}

impl rosidl_runtime_rs::Message for TwistWithCovarianceStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for TwistWithCovarianceStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/TwistWithCovarianceStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__TwistWithCovarianceStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Vector3()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Vector3__init(msg: *mut Vector3) -> bool;
    fn geometry_msgs__msg__Vector3__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Vector3>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Vector3__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Vector3>);
    fn geometry_msgs__msg__Vector3__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Vector3>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Vector3>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Vector3
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a vector in free space.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector3 {
    /// This is semantically different than a point.
    /// A vector is always anchored at the origin.
    /// When a transform is applied to a vector, only the rotational component is applied.
    pub x: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub y: f64,

    // This member is not documented.
    #[allow(missing_docs)]
    pub z: f64,
}

impl Default for Vector3 {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Vector3__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Vector3__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Vector3 {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Vector3 {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Vector3
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Vector3";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Vector3()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Vector3Stamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Vector3Stamped__init(msg: *mut Vector3Stamped) -> bool;
    fn geometry_msgs__msg__Vector3Stamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Vector3Stamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Vector3Stamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Vector3Stamped>,
    );
    fn geometry_msgs__msg__Vector3Stamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Vector3Stamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Vector3Stamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Vector3Stamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents a Vector3 with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector3Stamped {
    /// Note that this follows vector semantics with it always anchored at the origin,
    /// so the rotational elements of a transform are the only parts applied when transforming.
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub vector: super::super::msg::rmw::Vector3,
}

impl Default for Vector3Stamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Vector3Stamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Vector3Stamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Vector3Stamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3Stamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3Stamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Vector3Stamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Vector3Stamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Vector3Stamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Vector3Stamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Vector3Stamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__VelocityStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__VelocityStamped__init(msg: *mut VelocityStamped) -> bool;
    fn geometry_msgs__msg__VelocityStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<VelocityStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__VelocityStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<VelocityStamped>,
    );
    fn geometry_msgs__msg__VelocityStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<VelocityStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<VelocityStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__VelocityStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This expresses the timestamped velocity vector of a frame 'body_frame_id' in the reference frame 'reference_frame_id' expressed from arbitrary observation frame 'header.frame_id'.
/// - If the 'body_frame_id' and 'header.frame_id' are identical, the velocity is observed and defined in the local coordinates system of the body
///   which is the usual use-case in mobile robotics and is also known as a body twist.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VelocityStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub body_frame_id: rosidl_runtime_rs::String,

    // This member is not documented.
    #[allow(missing_docs)]
    pub reference_frame_id: rosidl_runtime_rs::String,

    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: super::super::msg::rmw::Twist,
}

impl Default for VelocityStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__VelocityStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__VelocityStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for VelocityStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__VelocityStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__VelocityStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__VelocityStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for VelocityStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for VelocityStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/VelocityStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__VelocityStamped()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Wrench()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__Wrench__init(msg: *mut Wrench) -> bool;
    fn geometry_msgs__msg__Wrench__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Wrench>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__Wrench__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Wrench>);
    fn geometry_msgs__msg__Wrench__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Wrench>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Wrench>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__Wrench
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This represents force in free space, separated into its linear and angular parts.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Wrench {
    // This member is not documented.
    #[allow(missing_docs)]
    pub force: super::super::msg::rmw::Vector3,

    // This member is not documented.
    #[allow(missing_docs)]
    pub torque: super::super::msg::rmw::Vector3,
}

impl Default for Wrench {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__Wrench__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__Wrench__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Wrench {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Wrench__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Wrench__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__Wrench__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Wrench {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Wrench
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/Wrench";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__Wrench()
        }
    }
}

#[link(name = "geometry_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__WrenchStamped()
    -> *const std::ffi::c_void;
}

#[link(name = "geometry_msgs__rosidl_generator_c")]
extern "C" {
    fn geometry_msgs__msg__WrenchStamped__init(msg: *mut WrenchStamped) -> bool;
    fn geometry_msgs__msg__WrenchStamped__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<WrenchStamped>,
        size: usize,
    ) -> bool;
    fn geometry_msgs__msg__WrenchStamped__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<WrenchStamped>,
    );
    fn geometry_msgs__msg__WrenchStamped__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<WrenchStamped>,
        out_seq: *mut rosidl_runtime_rs::Sequence<WrenchStamped>,
    ) -> bool;
}

// Corresponds to geometry_msgs__msg__WrenchStamped
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// A wrench with reference coordinate frame and timestamp

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WrenchStamped {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    // This member is not documented.
    #[allow(missing_docs)]
    pub wrench: super::super::msg::rmw::Wrench,
}

impl Default for WrenchStamped {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !geometry_msgs__msg__WrenchStamped__init(&mut msg as *mut _) {
                panic!("Call to geometry_msgs__msg__WrenchStamped__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for WrenchStamped {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__WrenchStamped__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__WrenchStamped__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { geometry_msgs__msg__WrenchStamped__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for WrenchStamped {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for WrenchStamped
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "geometry_msgs/msg/WrenchStamped";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__geometry_msgs__msg__WrenchStamped(
            )
        }
    }
}
