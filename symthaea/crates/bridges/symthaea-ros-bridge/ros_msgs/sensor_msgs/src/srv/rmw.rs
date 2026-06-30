#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__srv__SetCameraInfo_Request()
    -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__srv__SetCameraInfo_Request__init(msg: *mut SetCameraInfo_Request) -> bool;
    fn sensor_msgs__srv__SetCameraInfo_Request__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Request>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__srv__SetCameraInfo_Request__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Request>,
    );
    fn sensor_msgs__srv__SetCameraInfo_Request__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<SetCameraInfo_Request>,
        out_seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Request>,
    ) -> bool;
}

// Corresponds to sensor_msgs__srv__SetCameraInfo_Request
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
// This struct is not documented.
#[allow(missing_docs)]
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetCameraInfo_Request {
    /// The camera_info to store
    pub camera_info: super::super::msg::rmw::CameraInfo,
}

impl Default for SetCameraInfo_Request {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__srv__SetCameraInfo_Request__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__srv__SetCameraInfo_Request__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for SetCameraInfo_Request {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__srv__SetCameraInfo_Request__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__srv__SetCameraInfo_Request__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            sensor_msgs__srv__SetCameraInfo_Request__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for SetCameraInfo_Request {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for SetCameraInfo_Request
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/srv/SetCameraInfo_Request";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__srv__SetCameraInfo_Request()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__srv__SetCameraInfo_Response()
    -> *const std::ffi::c_void;
}

#[link(name = "sensor_msgs__rosidl_generator_c")]
extern "C" {
    fn sensor_msgs__srv__SetCameraInfo_Response__init(msg: *mut SetCameraInfo_Response) -> bool;
    fn sensor_msgs__srv__SetCameraInfo_Response__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Response>,
        size: usize,
    ) -> bool;
    fn sensor_msgs__srv__SetCameraInfo_Response__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Response>,
    );
    fn sensor_msgs__srv__SetCameraInfo_Response__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<SetCameraInfo_Response>,
        out_seq: *mut rosidl_runtime_rs::Sequence<SetCameraInfo_Response>,
    ) -> bool;
}

// Corresponds to sensor_msgs__srv__SetCameraInfo_Response
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
// This struct is not documented.
#[allow(missing_docs)]
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetCameraInfo_Response {
    /// True if the call succeeded
    pub success: bool,

    /// Used to give details about success
    pub status_message: rosidl_runtime_rs::String,
}

impl Default for SetCameraInfo_Response {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !sensor_msgs__srv__SetCameraInfo_Response__init(&mut msg as *mut _) {
                panic!("Call to sensor_msgs__srv__SetCameraInfo_Response__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for SetCameraInfo_Response {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__srv__SetCameraInfo_Response__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { sensor_msgs__srv__SetCameraInfo_Response__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            sensor_msgs__srv__SetCameraInfo_Response__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for SetCameraInfo_Response {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for SetCameraInfo_Response
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "sensor_msgs/srv/SetCameraInfo_Response";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__sensor_msgs__srv__SetCameraInfo_Response()
        }
    }
}

#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__sensor_msgs__srv__SetCameraInfo()
    -> *const std::ffi::c_void;
}

// Corresponds to sensor_msgs__srv__SetCameraInfo
#[allow(missing_docs, non_camel_case_types)]
pub struct SetCameraInfo;

impl rosidl_runtime_rs::Service for SetCameraInfo {
    type Request = SetCameraInfo_Request;
    type Response = SetCameraInfo_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_service_type_support_handle__sensor_msgs__srv__SetCameraInfo()
        }
    }
}
