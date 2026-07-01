#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to trajectory_msgs__msg__JointTrajectory
/// The header is used to specify the coordinate frame and the reference time for
/// the trajectory durations

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointTrajectory {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    /// The names of the active joints in each trajectory point. These names are
    /// ordered and must correspond to the values in each trajectory point.
    pub joint_names: Vec<std::string::String>,

    /// Array of trajectory points, which describe the positions, velocities,
    /// accelerations and/or efforts of the joints at each time point.
    pub points: Vec<super::msg::JointTrajectoryPoint>,

}



impl Default for JointTrajectory {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointTrajectory::default())
  }
}

impl rosidl_runtime_rs::Message for JointTrajectory {
  type RmwMsg = super::msg::rmw::JointTrajectory;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        joint_names: msg.joint_names
          .into_iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        points: msg.points
          .into_iter()
          .map(|elem| super::msg::JointTrajectoryPoint::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        joint_names: msg.joint_names
          .iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        points: msg.points
          .iter()
          .map(|elem| super::msg::JointTrajectoryPoint::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      joint_names: msg.joint_names
          .into_iter()
          .map(|elem| elem.to_string())
          .collect(),
      points: msg.points
          .into_iter()
          .map(super::msg::JointTrajectoryPoint::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to trajectory_msgs__msg__JointTrajectoryPoint
/// Each trajectory point specifies either positions[, velocities[, accelerations]]
/// or positions[, effort] for the trajectory to be executed.
/// All specified values are in the same order as the joint names in JointTrajectory.msg.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointTrajectoryPoint {
    /// Single DOF joint positions for each joint relative to their "0" position.
    /// The units depend on the specific joint type: radians for revolute or
    /// continuous joints, and meters for prismatic joints.
    pub positions: Vec<f64>,

    /// The rate of change in position of each joint. Units are joint type dependent.
    /// Radians/second for revolute or continuous joints, and meters/second for
    /// prismatic joints.
    pub velocities: Vec<f64>,

    /// Rate of change in velocity of each joint. Units are joint type dependent.
    /// Radians/second^2 for revolute or continuous joints, and meters/second^2 for
    /// prismatic joints.
    pub accelerations: Vec<f64>,

    /// The torque or the force to be applied at each joint. For revolute/continuous
    /// joints effort denotes a torque in newton-meters. For prismatic joints, effort
    /// denotes a force in newtons.
    pub effort: Vec<f64>,

    /// Desired time from the trajectory start to arrive at this trajectory point.
    pub time_from_start: builtin_interfaces::msg::Duration,

}



impl Default for JointTrajectoryPoint {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::JointTrajectoryPoint::default())
  }
}

impl rosidl_runtime_rs::Message for JointTrajectoryPoint {
  type RmwMsg = super::msg::rmw::JointTrajectoryPoint;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        positions: msg.positions.into(),
        velocities: msg.velocities.into(),
        accelerations: msg.accelerations.into(),
        effort: msg.effort.into(),
        time_from_start: builtin_interfaces::msg::Duration::into_rmw_message(std::borrow::Cow::Owned(msg.time_from_start)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        positions: msg.positions.as_slice().into(),
        velocities: msg.velocities.as_slice().into(),
        accelerations: msg.accelerations.as_slice().into(),
        effort: msg.effort.as_slice().into(),
        time_from_start: builtin_interfaces::msg::Duration::into_rmw_message(std::borrow::Cow::Borrowed(&msg.time_from_start)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      positions: msg.positions
          .into_iter()
          .collect(),
      velocities: msg.velocities
          .into_iter()
          .collect(),
      accelerations: msg.accelerations
          .into_iter()
          .collect(),
      effort: msg.effort
          .into_iter()
          .collect(),
      time_from_start: builtin_interfaces::msg::Duration::from_rmw_message(msg.time_from_start),
    }
  }
}


// Corresponds to trajectory_msgs__msg__MultiDOFJointTrajectory
/// The header is used to specify the coordinate frame and the reference time for the trajectory durations

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointTrajectory {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,

    /// A representation of a multi-dof joint trajectory (each point is a transformation)
    /// Each point along the trajectory will include an array of positions/velocities/accelerations
    /// that has the same length as the array of joint names, and has the same order of joints as
    /// the joint names array.
    pub joint_names: Vec<std::string::String>,


    // This member is not documented.
    #[allow(missing_docs)]
    pub points: Vec<super::msg::MultiDOFJointTrajectoryPoint>,

}



impl Default for MultiDOFJointTrajectory {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MultiDOFJointTrajectory::default())
  }
}

impl rosidl_runtime_rs::Message for MultiDOFJointTrajectory {
  type RmwMsg = super::msg::rmw::MultiDOFJointTrajectory;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        joint_names: msg.joint_names
          .into_iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        points: msg.points
          .into_iter()
          .map(|elem| super::msg::MultiDOFJointTrajectoryPoint::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        joint_names: msg.joint_names
          .iter()
          .map(|elem| elem.as_str().into())
          .collect(),
        points: msg.points
          .iter()
          .map(|elem| super::msg::MultiDOFJointTrajectoryPoint::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      joint_names: msg.joint_names
          .into_iter()
          .map(|elem| elem.to_string())
          .collect(),
      points: msg.points
          .into_iter()
          .map(super::msg::MultiDOFJointTrajectoryPoint::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to trajectory_msgs__msg__MultiDOFJointTrajectoryPoint
/// Each multi-dof joint can specify a transform (up to 6 DOF).

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointTrajectoryPoint {

    // This member is not documented.
    #[allow(missing_docs)]
    pub transforms: Vec<geometry_msgs::msg::Transform>,

    /// There can be a velocity specified for the origin of the joint.
    pub velocities: Vec<geometry_msgs::msg::Twist>,

    /// There can be an acceleration specified for the origin of the joint.
    pub accelerations: Vec<geometry_msgs::msg::Twist>,

    /// Desired time from the trajectory start to arrive at this trajectory point.
    pub time_from_start: builtin_interfaces::msg::Duration,

}



impl Default for MultiDOFJointTrajectoryPoint {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::MultiDOFJointTrajectoryPoint::default())
  }
}

impl rosidl_runtime_rs::Message for MultiDOFJointTrajectoryPoint {
  type RmwMsg = super::msg::rmw::MultiDOFJointTrajectoryPoint;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        transforms: msg.transforms
          .into_iter()
          .map(|elem| geometry_msgs::msg::Transform::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        velocities: msg.velocities
          .into_iter()
          .map(|elem| geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        accelerations: msg.accelerations
          .into_iter()
          .map(|elem| geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
        time_from_start: builtin_interfaces::msg::Duration::into_rmw_message(std::borrow::Cow::Owned(msg.time_from_start)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        transforms: msg.transforms
          .iter()
          .map(|elem| geometry_msgs::msg::Transform::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
        velocities: msg.velocities
          .iter()
          .map(|elem| geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
        accelerations: msg.accelerations
          .iter()
          .map(|elem| geometry_msgs::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
        time_from_start: builtin_interfaces::msg::Duration::into_rmw_message(std::borrow::Cow::Borrowed(&msg.time_from_start)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      transforms: msg.transforms
          .into_iter()
          .map(geometry_msgs::msg::Transform::from_rmw_message)
          .collect(),
      velocities: msg.velocities
          .into_iter()
          .map(geometry_msgs::msg::Twist::from_rmw_message)
          .collect(),
      accelerations: msg.accelerations
          .into_iter()
          .map(geometry_msgs::msg::Twist::from_rmw_message)
          .collect(),
      time_from_start: builtin_interfaces::msg::Duration::from_rmw_message(msg.time_from_start),
    }
  }
}


