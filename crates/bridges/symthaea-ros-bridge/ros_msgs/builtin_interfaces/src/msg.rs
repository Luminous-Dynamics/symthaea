#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to builtin_interfaces__msg__Duration
/// Duration defines a period between two time points.
/// Messages of this datatype are of ROS Time following this design:
/// https://design.ros2.org/articles/clock_and_time.html

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Duration::default())
  }
}

impl rosidl_runtime_rs::Message for Duration {
  type RmwMsg = super::msg::rmw::Duration;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        sec: msg.sec,
        nanosec: msg.nanosec,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      sec: msg.sec,
      nanosec: msg.nanosec,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      sec: msg.sec,
      nanosec: msg.nanosec,
    }
  }
}


// Corresponds to builtin_interfaces__msg__Time
/// This message communicates ROS Time defined here:
/// https://design.ros2.org/articles/clock_and_time.html

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Time::default())
  }
}

impl rosidl_runtime_rs::Message for Time {
  type RmwMsg = super::msg::rmw::Time;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        sec: msg.sec,
        nanosec: msg.nanosec,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      sec: msg.sec,
      nanosec: msg.nanosec,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      sec: msg.sec,
      nanosec: msg.nanosec,
    }
  }
}


