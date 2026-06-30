#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[link(name = "trajectory_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__JointTrajectory()
    -> *const std::ffi::c_void;
}

#[link(name = "trajectory_msgs__rosidl_generator_c")]
extern "C" {
    fn trajectory_msgs__msg__JointTrajectory__init(msg: *mut JointTrajectory) -> bool;
    fn trajectory_msgs__msg__JointTrajectory__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<JointTrajectory>,
        size: usize,
    ) -> bool;
    fn trajectory_msgs__msg__JointTrajectory__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<JointTrajectory>,
    );
    fn trajectory_msgs__msg__JointTrajectory__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<JointTrajectory>,
        out_seq: *mut rosidl_runtime_rs::Sequence<JointTrajectory>,
    ) -> bool;
}

// Corresponds to trajectory_msgs__msg__JointTrajectory
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// The header is used to specify the coordinate frame and the reference time for
/// the trajectory durations

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointTrajectory {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    /// The names of the active joints in each trajectory point. These names are
    /// ordered and must correspond to the values in each trajectory point.
    pub joint_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    /// Array of trajectory points, which describe the positions, velocities,
    /// accelerations and/or efforts of the joints at each time point.
    pub points: rosidl_runtime_rs::Sequence<super::super::msg::rmw::JointTrajectoryPoint>,
}

impl Default for JointTrajectory {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !trajectory_msgs__msg__JointTrajectory__init(&mut msg as *mut _) {
                panic!("Call to trajectory_msgs__msg__JointTrajectory__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for JointTrajectory {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__JointTrajectory__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__JointTrajectory__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__JointTrajectory__Sequence__copy(in_seq, out_seq as *mut _) }
    }
}

impl rosidl_runtime_rs::Message for JointTrajectory {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for JointTrajectory
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "trajectory_msgs/msg/JointTrajectory";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__JointTrajectory()
        }
    }
}

#[link(name = "trajectory_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__JointTrajectoryPoint()
    -> *const std::ffi::c_void;
}

#[link(name = "trajectory_msgs__rosidl_generator_c")]
extern "C" {
    fn trajectory_msgs__msg__JointTrajectoryPoint__init(msg: *mut JointTrajectoryPoint) -> bool;
    fn trajectory_msgs__msg__JointTrajectoryPoint__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<JointTrajectoryPoint>,
        size: usize,
    ) -> bool;
    fn trajectory_msgs__msg__JointTrajectoryPoint__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<JointTrajectoryPoint>,
    );
    fn trajectory_msgs__msg__JointTrajectoryPoint__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<JointTrajectoryPoint>,
        out_seq: *mut rosidl_runtime_rs::Sequence<JointTrajectoryPoint>,
    ) -> bool;
}

// Corresponds to trajectory_msgs__msg__JointTrajectoryPoint
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Each trajectory point specifies either positions[, velocities[, accelerations]]
/// or positions[, effort] for the trajectory to be executed.
/// All specified values are in the same order as the joint names in JointTrajectory.msg.

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct JointTrajectoryPoint {
    /// Single DOF joint positions for each joint relative to their "0" position.
    /// The units depend on the specific joint type: radians for revolute or
    /// continuous joints, and meters for prismatic joints.
    pub positions: rosidl_runtime_rs::Sequence<f64>,

    /// The rate of change in position of each joint. Units are joint type dependent.
    /// Radians/second for revolute or continuous joints, and meters/second for
    /// prismatic joints.
    pub velocities: rosidl_runtime_rs::Sequence<f64>,

    /// Rate of change in velocity of each joint. Units are joint type dependent.
    /// Radians/second^2 for revolute or continuous joints, and meters/second^2 for
    /// prismatic joints.
    pub accelerations: rosidl_runtime_rs::Sequence<f64>,

    /// The torque or the force to be applied at each joint. For revolute/continuous
    /// joints effort denotes a torque in newton-meters. For prismatic joints, effort
    /// denotes a force in newtons.
    pub effort: rosidl_runtime_rs::Sequence<f64>,

    /// Desired time from the trajectory start to arrive at this trajectory point.
    pub time_from_start: builtin_interfaces::msg::rmw::Duration,
}

impl Default for JointTrajectoryPoint {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !trajectory_msgs__msg__JointTrajectoryPoint__init(&mut msg as *mut _) {
                panic!("Call to trajectory_msgs__msg__JointTrajectoryPoint__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for JointTrajectoryPoint {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__JointTrajectoryPoint__Sequence__init(seq as *mut _, size) }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__JointTrajectoryPoint__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            trajectory_msgs__msg__JointTrajectoryPoint__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for JointTrajectoryPoint {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for JointTrajectoryPoint
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "trajectory_msgs/msg/JointTrajectoryPoint";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__JointTrajectoryPoint()
        }
    }
}

#[link(name = "trajectory_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__MultiDOFJointTrajectory()
    -> *const std::ffi::c_void;
}

#[link(name = "trajectory_msgs__rosidl_generator_c")]
extern "C" {
    fn trajectory_msgs__msg__MultiDOFJointTrajectory__init(
        msg: *mut MultiDOFJointTrajectory,
    ) -> bool;
    fn trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectory>,
        size: usize,
    ) -> bool;
    fn trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectory>,
    );
    fn trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<MultiDOFJointTrajectory>,
        out_seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectory>,
    ) -> bool;
}

// Corresponds to trajectory_msgs__msg__MultiDOFJointTrajectory
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// The header is used to specify the coordinate frame and the reference time for the trajectory durations

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointTrajectory {
    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::rmw::Header,

    /// A representation of a multi-dof joint trajectory (each point is a transformation)
    /// Each point along the trajectory will include an array of positions/velocities/accelerations
    /// that has the same length as the array of joint names, and has the same order of joints as
    /// the joint names array.
    pub joint_names: rosidl_runtime_rs::Sequence<rosidl_runtime_rs::String>,

    // This member is not documented.
    #[allow(missing_docs)]
    pub points: rosidl_runtime_rs::Sequence<super::super::msg::rmw::MultiDOFJointTrajectoryPoint>,
}

impl Default for MultiDOFJointTrajectory {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !trajectory_msgs__msg__MultiDOFJointTrajectory__init(&mut msg as *mut _) {
                panic!("Call to trajectory_msgs__msg__MultiDOFJointTrajectory__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for MultiDOFJointTrajectory {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__init(seq as *mut _, size)
        }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            trajectory_msgs__msg__MultiDOFJointTrajectory__Sequence__copy(in_seq, out_seq as *mut _)
        }
    }
}

impl rosidl_runtime_rs::Message for MultiDOFJointTrajectory {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for MultiDOFJointTrajectory
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "trajectory_msgs/msg/MultiDOFJointTrajectory";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__MultiDOFJointTrajectory()
        }
    }
}

#[link(name = "trajectory_msgs__rosidl_typesupport_c")]
extern "C" {
    fn rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__MultiDOFJointTrajectoryPoint()
    -> *const std::ffi::c_void;
}

#[link(name = "trajectory_msgs__rosidl_generator_c")]
extern "C" {
    fn trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__init(
        msg: *mut MultiDOFJointTrajectoryPoint,
    ) -> bool;
    fn trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__init(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectoryPoint>,
        size: usize,
    ) -> bool;
    fn trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__fini(
        seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectoryPoint>,
    );
    fn trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__copy(
        in_seq: &rosidl_runtime_rs::Sequence<MultiDOFJointTrajectoryPoint>,
        out_seq: *mut rosidl_runtime_rs::Sequence<MultiDOFJointTrajectoryPoint>,
    ) -> bool;
}

// Corresponds to trajectory_msgs__msg__MultiDOFJointTrajectoryPoint
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]

/// Each multi-dof joint can specify a transform (up to 6 DOF).

#[repr(C)]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct MultiDOFJointTrajectoryPoint {
    // This member is not documented.
    #[allow(missing_docs)]
    pub transforms: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Transform>,

    /// There can be a velocity specified for the origin of the joint.
    pub velocities: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Twist>,

    /// There can be an acceleration specified for the origin of the joint.
    pub accelerations: rosidl_runtime_rs::Sequence<geometry_msgs::msg::rmw::Twist>,

    /// Desired time from the trajectory start to arrive at this trajectory point.
    pub time_from_start: builtin_interfaces::msg::rmw::Duration,
}

impl Default for MultiDOFJointTrajectoryPoint {
    fn default() -> Self {
        unsafe {
            let mut msg = std::mem::zeroed();
            if !trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__init(&mut msg as *mut _) {
                panic!("Call to trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__init() failed");
            }
            msg
        }
    }
}

impl rosidl_runtime_rs::SequenceAlloc for MultiDOFJointTrajectoryPoint {
    fn sequence_init(seq: &mut rosidl_runtime_rs::Sequence<Self>, size: usize) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__init(seq as *mut _, size)
        }
    }
    fn sequence_fini(seq: &mut rosidl_runtime_rs::Sequence<Self>) {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe { trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__fini(seq as *mut _) }
    }
    fn sequence_copy(
        in_seq: &rosidl_runtime_rs::Sequence<Self>,
        out_seq: &mut rosidl_runtime_rs::Sequence<Self>,
    ) -> bool {
        // SAFETY: This is safe since the pointer is guaranteed to be valid/initialized.
        unsafe {
            trajectory_msgs__msg__MultiDOFJointTrajectoryPoint__Sequence__copy(
                in_seq,
                out_seq as *mut _,
            )
        }
    }
}

impl rosidl_runtime_rs::Message for MultiDOFJointTrajectoryPoint {
    type RmwMsg = Self;
    fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
        msg_cow
    }
    fn from_rmw_message(msg: Self::RmwMsg) -> Self {
        msg
    }
}

impl rosidl_runtime_rs::RmwMessage for MultiDOFJointTrajectoryPoint
where
    Self: Sized,
{
    const TYPE_NAME: &'static str = "trajectory_msgs/msg/MultiDOFJointTrajectoryPoint";
    fn get_type_support() -> *const std::ffi::c_void {
        // SAFETY: No preconditions for this function.
        unsafe {
            rosidl_typesupport_c__get_message_type_support_handle__trajectory_msgs__msg__MultiDOFJointTrajectoryPoint()
        }
    }
}
