#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "builtin_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__builtin_interfaces__msg__Duration()
    -> *const std::ffi::c_void;
}

#[link(name = "builtin_interfaces__rosidl_generator_c")]
extern "C" {
    fn builtin_interfaces__msg__Duration__init(msg: *mut Duration) -> bool;
    fn builtin_interfaces__msg__Duration__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Duration>,
        size: usize,
    ) -> bool;
    fn builtin_interfaces__msg__Duration__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<Duration>,
    );
    fn builtin_interfaces__msg__Duration__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Duration>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Duration>,
    ) -> bool;
}

// Corresponds to builtin_interfaces__msg__Duration
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Duration defines a period between two time points.
/// Messages of this datatype are of ROS Time following this design:
/// https://design.ros2.org/articles/clock_and_time.html

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Duration {
    /// The seconds component, valid over all int32 values.
    pub sec: i32,

    /// The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
    /// e.g.
    /// The duration -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
    /// The duration 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
    pub nanosec: u32,
}

impl Default for Duration {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !builtin_interfaces__msg__Duration__init(&mut msg as *mut _) {
                panic!("Call to builtin_interfaces__msg__Duration__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Duration {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Duration__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Duration__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Duration__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Duration {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Duration
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "builtin_interfaces/msg/Duration";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__builtin_interfaces__msg__Duration(
            )
        }
    }
}

#[link(name = "builtin_interfaces__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__builtin_interfaces__msg__Time()
    -> *const std::ffi::c_void;
}

#[link(name = "builtin_interfaces__rosidl_generator_c")]
extern "C" {
    fn builtin_interfaces__msg__Time__init(msg: *mut Time) -> bool;
    fn builtin_interfaces__msg__Time__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<Time>,
        size: usize,
    ) -> bool;
    fn builtin_interfaces__msg__Time__Sequence__fini(seq: *mut rosidl_runtime_rs::Sequence<Time>);
    fn builtin_interfaces__msg__Time__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<Time>,
        out_seq: *mut rosidl_runtime_rs::Sequence<Time>,
    ) -> bool;
}

// Corresponds to builtin_interfaces__msg__Time
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// This message communicates ROS Time defined here:
/// https://design.ros2.org/articles/clock_and_time.html

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Time {
    /// The seconds component, valid over all int32 values.
    pub sec: i32,

    /// The nanoseconds component, valid in the range [0, 1e9), to be added to the seconds component.
    /// e.g.
    /// The time -1.7 seconds is represented as {sec: -2, nanosec: 3e8}
    /// The time 1.7 seconds is represented as {sec: 1, nanosec: 7e8}
    pub nanosec: u32,
}

impl Default for Time {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !builtin_interfaces__msg__Time__init(&mut msg as *mut _) {
                panic!("Call to builtin_interfaces__msg__Time__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for Time {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Time__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Time__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { builtin_interfaces__msg__Time__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for Time {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for Time
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "builtin_interfaces/msg/Time";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__builtin_interfaces__msg__Time()
        }
    }
}
