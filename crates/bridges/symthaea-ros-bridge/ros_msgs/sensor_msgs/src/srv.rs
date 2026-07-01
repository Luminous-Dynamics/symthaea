#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};




// Corresponds to sensor_msgs__srv__SetCameraInfo_Request

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetCameraInfo_Request {
    /// The camera_info to store
    pub camera_info: super::msg::CameraInfo,

}



impl Default for SetCameraInfo_Request {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetCameraInfo_Request::default())
  }
}

impl rosidl_runtime_rs::Message for SetCameraInfo_Request {
  type RmwMsg = super::srv::rmw::SetCameraInfo_Request;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        camera_info: super::msg::CameraInfo::into_rmw_message(std::borrow::Cow::Owned(msg.camera_info)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        camera_info: super::msg::CameraInfo::into_rmw_message(std::borrow::Cow::Borrowed(&msg.camera_info)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      camera_info: super::msg::CameraInfo::from_rmw_message(msg.camera_info),
    }
  }
}


// Corresponds to sensor_msgs__srv__SetCameraInfo_Response

// This struct is not documented.
#[allow(missing_docs)]

#[allow(non_camel_case_types)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct SetCameraInfo_Response {
    /// True if the call succeeded
    pub success: bool,

    /// Used to give details about success
    pub status_message: std::string::String,

}



impl Default for SetCameraInfo_Response {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::srv::rmw::SetCameraInfo_Response::default())
  }
}

impl rosidl_runtime_rs::Message for SetCameraInfo_Response {
  type RmwMsg = super::srv::rmw::SetCameraInfo_Response;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        success: msg.success,
        status_message: msg.status_message.as_str().into(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      success: msg.success,
        status_message: msg.status_message.as_str().into(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      success: msg.success,
      status_message: msg.status_message.to_string(),
    }
  }
}






#[link(name = "sensor_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_service_type_support_handle__sensor_msgs__srv__SetCameraInfo() -> *const std::ffi::c_void;
}

// Corresponds to sensor_msgs__srv__SetCameraInfo
#[allow(missing_docs, non_camel_case_types)]
pub struct SetCameraInfo;

impl rosidl_runtime_rs::Service for SetCameraInfo {
    type Request = SetCameraInfo_Request;
    type Response = SetCameraInfo_Response;

    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe { rosidl_typesupport_c__get_service_type_support_handle__sensor_msgs__srv__SetCameraInfo() }
    }
}


