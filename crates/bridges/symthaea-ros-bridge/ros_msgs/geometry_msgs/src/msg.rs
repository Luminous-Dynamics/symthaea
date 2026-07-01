#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};



// Corresponds to geometry_msgs__msg__Accel
/// This expresses acceleration in free space broken into its linear and angular parts.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Accel {

    // This member is not documented.
    #[allow(missing_docs)]
    pub linear: super::msg::Vector3,


    // This member is not documented.
    #[allow(missing_docs)]
    pub angular: super::msg::Vector3,

}



impl Default for Accel {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Accel::default())
  }
}

impl rosidl_runtime_rs::Message for Accel {
  type RmwMsg = super::msg::rmw::Accel;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        linear: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.linear)).into_owned(),
        angular: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.angular)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        linear: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.linear)).into_owned(),
        angular: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.angular)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      linear: super::msg::Vector3::from_rmw_message(msg.linear),
      angular: super::msg::Vector3::from_rmw_message(msg.angular),
    }
  }
}


// Corresponds to geometry_msgs__msg__AccelStamped
/// An accel with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::msg::Accel,

}



impl Default for AccelStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::AccelStamped::default())
  }
}

impl rosidl_runtime_rs::Message for AccelStamped {
  type RmwMsg = super::msg::rmw::AccelStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        accel: super::msg::Accel::into_rmw_message(std::borrow::Cow::Owned(msg.accel)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        accel: super::msg::Accel::into_rmw_message(std::borrow::Cow::Borrowed(&msg.accel)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      accel: super::msg::Accel::from_rmw_message(msg.accel),
    }
  }
}


// Corresponds to geometry_msgs__msg__AccelWithCovariance
/// This expresses acceleration in free space with uncertainty.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelWithCovariance {

    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::msg::Accel,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],

}



impl Default for AccelWithCovariance {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::AccelWithCovariance::default())
  }
}

impl rosidl_runtime_rs::Message for AccelWithCovariance {
  type RmwMsg = super::msg::rmw::AccelWithCovariance;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        accel: super::msg::Accel::into_rmw_message(std::borrow::Cow::Owned(msg.accel)).into_owned(),
        covariance: msg.covariance,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        accel: super::msg::Accel::into_rmw_message(std::borrow::Cow::Borrowed(&msg.accel)).into_owned(),
        covariance: msg.covariance,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      accel: super::msg::Accel::from_rmw_message(msg.accel),
      covariance: msg.covariance,
    }
  }
}


// Corresponds to geometry_msgs__msg__AccelWithCovarianceStamped
/// This represents an estimated accel with reference coordinate frame and timestamp.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct AccelWithCovarianceStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub accel: super::msg::AccelWithCovariance,

}



impl Default for AccelWithCovarianceStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::AccelWithCovarianceStamped::default())
  }
}

impl rosidl_runtime_rs::Message for AccelWithCovarianceStamped {
  type RmwMsg = super::msg::rmw::AccelWithCovarianceStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        accel: super::msg::AccelWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.accel)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        accel: super::msg::AccelWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.accel)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      accel: super::msg::AccelWithCovariance::from_rmw_message(msg.accel),
    }
  }
}


// Corresponds to geometry_msgs__msg__Inertia
/// Mass

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Inertia {

    // This member is not documented.
    #[allow(missing_docs)]
    pub m: f64,

    /// Center of mass
    pub com: super::msg::Vector3,

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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Inertia::default())
  }
}

impl rosidl_runtime_rs::Message for Inertia {
  type RmwMsg = super::msg::rmw::Inertia;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        m: msg.m,
        com: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.com)).into_owned(),
        ixx: msg.ixx,
        ixy: msg.ixy,
        ixz: msg.ixz,
        iyy: msg.iyy,
        iyz: msg.iyz,
        izz: msg.izz,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      m: msg.m,
        com: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.com)).into_owned(),
      ixx: msg.ixx,
      ixy: msg.ixy,
      ixz: msg.ixz,
      iyy: msg.iyy,
      iyz: msg.iyz,
      izz: msg.izz,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      m: msg.m,
      com: super::msg::Vector3::from_rmw_message(msg.com),
      ixx: msg.ixx,
      ixy: msg.ixy,
      ixz: msg.ixz,
      iyy: msg.iyy,
      iyz: msg.iyz,
      izz: msg.izz,
    }
  }
}


// Corresponds to geometry_msgs__msg__InertiaStamped
/// An Inertia with a time stamp and reference frame.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct InertiaStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub inertia: super::msg::Inertia,

}



impl Default for InertiaStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::InertiaStamped::default())
  }
}

impl rosidl_runtime_rs::Message for InertiaStamped {
  type RmwMsg = super::msg::rmw::InertiaStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        inertia: super::msg::Inertia::into_rmw_message(std::borrow::Cow::Owned(msg.inertia)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        inertia: super::msg::Inertia::into_rmw_message(std::borrow::Cow::Borrowed(&msg.inertia)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      inertia: super::msg::Inertia::from_rmw_message(msg.inertia),
    }
  }
}


// Corresponds to geometry_msgs__msg__Point
/// This contains the position of a point in free space

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Point::default())
  }
}

impl rosidl_runtime_rs::Message for Point {
  type RmwMsg = super::msg::rmw::Point;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        z: msg.z,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      z: msg.z,
    }
  }
}


// Corresponds to geometry_msgs__msg__Point32
/// This contains the position of a point in free space(with 32 bits of precision).
/// It is recommended to use Point wherever possible instead of Point32.
///
/// This recommendation is to promote interoperability.
///
/// This message is designed to take up less space when sending
/// lots of points at once, as in the case of a PointCloud.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Point32::default())
  }
}

impl rosidl_runtime_rs::Message for Point32 {
  type RmwMsg = super::msg::rmw::Point32;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        z: msg.z,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      z: msg.z,
    }
  }
}


// Corresponds to geometry_msgs__msg__PointStamped
/// This represents a Point with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PointStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub point: super::msg::Point,

}



impl Default for PointStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PointStamped::default())
  }
}

impl rosidl_runtime_rs::Message for PointStamped {
  type RmwMsg = super::msg::rmw::PointStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        point: super::msg::Point::into_rmw_message(std::borrow::Cow::Owned(msg.point)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        point: super::msg::Point::into_rmw_message(std::borrow::Cow::Borrowed(&msg.point)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      point: super::msg::Point::from_rmw_message(msg.point),
    }
  }
}


// Corresponds to geometry_msgs__msg__Polygon
/// A specification of a polygon where the first and last points are assumed to be connected

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Polygon {

    // This member is not documented.
    #[allow(missing_docs)]
    pub points: Vec<super::msg::Point32>,

}



impl Default for Polygon {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Polygon::default())
  }
}

impl rosidl_runtime_rs::Message for Polygon {
  type RmwMsg = super::msg::rmw::Polygon;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        points: msg.points
          .into_iter()
          .map(|elem| super::msg::Point32::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        points: msg.points
          .iter()
          .map(|elem| super::msg::Point32::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      points: msg.points
          .into_iter()
          .map(super::msg::Point32::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to geometry_msgs__msg__PolygonInstance
/// A specification of a polygon where the first and last points are assumed to be connected
/// It includes a unique identification field for disambiguating multiple instances

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonInstance {

    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::msg::Polygon,


    // This member is not documented.
    #[allow(missing_docs)]
    pub id: i64,

}



impl Default for PolygonInstance {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PolygonInstance::default())
  }
}

impl rosidl_runtime_rs::Message for PolygonInstance {
  type RmwMsg = super::msg::rmw::PolygonInstance;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        polygon: super::msg::Polygon::into_rmw_message(std::borrow::Cow::Owned(msg.polygon)).into_owned(),
        id: msg.id,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        polygon: super::msg::Polygon::into_rmw_message(std::borrow::Cow::Borrowed(&msg.polygon)).into_owned(),
      id: msg.id,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      polygon: super::msg::Polygon::from_rmw_message(msg.polygon),
      id: msg.id,
    }
  }
}


// Corresponds to geometry_msgs__msg__PolygonInstanceStamped
/// This represents a Polygon with reference coordinate frame and timestamp
/// It includes a unique identification field for disambiguating multiple instances

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonInstanceStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::msg::PolygonInstance,

}



impl Default for PolygonInstanceStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PolygonInstanceStamped::default())
  }
}

impl rosidl_runtime_rs::Message for PolygonInstanceStamped {
  type RmwMsg = super::msg::rmw::PolygonInstanceStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        polygon: super::msg::PolygonInstance::into_rmw_message(std::borrow::Cow::Owned(msg.polygon)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        polygon: super::msg::PolygonInstance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.polygon)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      polygon: super::msg::PolygonInstance::from_rmw_message(msg.polygon),
    }
  }
}


// Corresponds to geometry_msgs__msg__PolygonStamped
/// This represents a Polygon with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PolygonStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub polygon: super::msg::Polygon,

}



impl Default for PolygonStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PolygonStamped::default())
  }
}

impl rosidl_runtime_rs::Message for PolygonStamped {
  type RmwMsg = super::msg::rmw::PolygonStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        polygon: super::msg::Polygon::into_rmw_message(std::borrow::Cow::Owned(msg.polygon)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        polygon: super::msg::Polygon::into_rmw_message(std::borrow::Cow::Borrowed(&msg.polygon)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      polygon: super::msg::Polygon::from_rmw_message(msg.polygon),
    }
  }
}


// Corresponds to geometry_msgs__msg__Pose
/// A representation of pose in free space, composed of position and orientation.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Pose {

    // This member is not documented.
    #[allow(missing_docs)]
    pub position: super::msg::Point,


    // This member is not documented.
    #[allow(missing_docs)]
    pub orientation: super::msg::Quaternion,

}



impl Default for Pose {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Pose::default())
  }
}

impl rosidl_runtime_rs::Message for Pose {
  type RmwMsg = super::msg::rmw::Pose;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        position: super::msg::Point::into_rmw_message(std::borrow::Cow::Owned(msg.position)).into_owned(),
        orientation: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Owned(msg.orientation)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        position: super::msg::Point::into_rmw_message(std::borrow::Cow::Borrowed(&msg.position)).into_owned(),
        orientation: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Borrowed(&msg.orientation)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      position: super::msg::Point::from_rmw_message(msg.position),
      orientation: super::msg::Quaternion::from_rmw_message(msg.orientation),
    }
  }
}


// Corresponds to geometry_msgs__msg__Pose2D
/// Deprecated as of Foxy and will potentially be removed in any following release.
/// Please use the full 3D pose.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Pose2D::default())
  }
}

impl rosidl_runtime_rs::Message for Pose2D {
  type RmwMsg = super::msg::rmw::Pose2D;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        theta: msg.theta,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      theta: msg.theta,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      theta: msg.theta,
    }
  }
}


// Corresponds to geometry_msgs__msg__PoseArray
/// An array of poses with a header for global reference.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseArray {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub poses: Vec<super::msg::Pose>,

}



impl Default for PoseArray {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PoseArray::default())
  }
}

impl rosidl_runtime_rs::Message for PoseArray {
  type RmwMsg = super::msg::rmw::PoseArray;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        poses: msg.poses
          .into_iter()
          .map(|elem| super::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(elem)).into_owned())
          .collect(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        poses: msg.poses
          .iter()
          .map(|elem| super::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(elem)).into_owned())
          .collect(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      poses: msg.poses
          .into_iter()
          .map(super::msg::Pose::from_rmw_message)
          .collect(),
    }
  }
}


// Corresponds to geometry_msgs__msg__PoseStamped
/// A Pose with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::msg::Pose,

}



impl Default for PoseStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PoseStamped::default())
  }
}

impl rosidl_runtime_rs::Message for PoseStamped {
  type RmwMsg = super::msg::rmw::PoseStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        pose: super::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        pose: super::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      pose: super::msg::Pose::from_rmw_message(msg.pose),
    }
  }
}


// Corresponds to geometry_msgs__msg__PoseWithCovariance
/// This represents a pose in free space with uncertainty.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseWithCovariance {

    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::msg::Pose,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],

}



impl Default for PoseWithCovariance {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PoseWithCovariance::default())
  }
}

impl rosidl_runtime_rs::Message for PoseWithCovariance {
  type RmwMsg = super::msg::rmw::PoseWithCovariance;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        pose: super::msg::Pose::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
        covariance: msg.covariance,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        pose: super::msg::Pose::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
        covariance: msg.covariance,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      pose: super::msg::Pose::from_rmw_message(msg.pose),
      covariance: msg.covariance,
    }
  }
}


// Corresponds to geometry_msgs__msg__PoseWithCovarianceStamped
/// This expresses an estimated pose with a reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct PoseWithCovarianceStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub pose: super::msg::PoseWithCovariance,

}



impl Default for PoseWithCovarianceStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::PoseWithCovarianceStamped::default())
  }
}

impl rosidl_runtime_rs::Message for PoseWithCovarianceStamped {
  type RmwMsg = super::msg::rmw::PoseWithCovarianceStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        pose: super::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.pose)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        pose: super::msg::PoseWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.pose)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      pose: super::msg::PoseWithCovariance::from_rmw_message(msg.pose),
    }
  }
}


// Corresponds to geometry_msgs__msg__Quaternion
/// This represents an orientation in free space in quaternion form.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Quaternion::default())
  }
}

impl rosidl_runtime_rs::Message for Quaternion {
  type RmwMsg = super::msg::rmw::Quaternion;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        z: msg.z,
        w: msg.w,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      w: msg.w,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      w: msg.w,
    }
  }
}


// Corresponds to geometry_msgs__msg__QuaternionStamped
/// This represents an orientation with reference coordinate frame and timestamp.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct QuaternionStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub quaternion: super::msg::Quaternion,

}



impl Default for QuaternionStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::QuaternionStamped::default())
  }
}

impl rosidl_runtime_rs::Message for QuaternionStamped {
  type RmwMsg = super::msg::rmw::QuaternionStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        quaternion: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Owned(msg.quaternion)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        quaternion: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Borrowed(&msg.quaternion)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      quaternion: super::msg::Quaternion::from_rmw_message(msg.quaternion),
    }
  }
}


// Corresponds to geometry_msgs__msg__Transform
/// This represents the transform between two coordinate frames in free space.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Transform {

    // This member is not documented.
    #[allow(missing_docs)]
    pub translation: super::msg::Vector3,


    // This member is not documented.
    #[allow(missing_docs)]
    pub rotation: super::msg::Quaternion,

}



impl Default for Transform {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Transform::default())
  }
}

impl rosidl_runtime_rs::Message for Transform {
  type RmwMsg = super::msg::rmw::Transform;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        translation: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.translation)).into_owned(),
        rotation: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Owned(msg.rotation)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        translation: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.translation)).into_owned(),
        rotation: super::msg::Quaternion::into_rmw_message(std::borrow::Cow::Borrowed(&msg.rotation)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      translation: super::msg::Vector3::from_rmw_message(msg.translation),
      rotation: super::msg::Quaternion::from_rmw_message(msg.rotation),
    }
  }
}


// Corresponds to geometry_msgs__msg__TransformStamped
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

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TransformStamped {
    /// The frame id in the header is used as the reference frame of this transform.
    pub header: std_msgs::msg::Header,

    /// The frame id of the child frame to which this transform points.
    pub child_frame_id: std::string::String,

    /// Translation and rotation in 3-dimensions of child_frame_id from header.frame_id.
    pub transform: super::msg::Transform,

}



impl Default for TransformStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::TransformStamped::default())
  }
}

impl rosidl_runtime_rs::Message for TransformStamped {
  type RmwMsg = super::msg::rmw::TransformStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        child_frame_id: msg.child_frame_id.as_str().into(),
        transform: super::msg::Transform::into_rmw_message(std::borrow::Cow::Owned(msg.transform)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        child_frame_id: msg.child_frame_id.as_str().into(),
        transform: super::msg::Transform::into_rmw_message(std::borrow::Cow::Borrowed(&msg.transform)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      child_frame_id: msg.child_frame_id.to_string(),
      transform: super::msg::Transform::from_rmw_message(msg.transform),
    }
  }
}


// Corresponds to geometry_msgs__msg__Twist
/// This expresses velocity in free space broken into its linear and angular parts.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Twist {

    // This member is not documented.
    #[allow(missing_docs)]
    pub linear: super::msg::Vector3,


    // This member is not documented.
    #[allow(missing_docs)]
    pub angular: super::msg::Vector3,

}



impl Default for Twist {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Twist::default())
  }
}

impl rosidl_runtime_rs::Message for Twist {
  type RmwMsg = super::msg::rmw::Twist;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        linear: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.linear)).into_owned(),
        angular: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.angular)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        linear: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.linear)).into_owned(),
        angular: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.angular)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      linear: super::msg::Vector3::from_rmw_message(msg.linear),
      angular: super::msg::Vector3::from_rmw_message(msg.angular),
    }
  }
}


// Corresponds to geometry_msgs__msg__TwistStamped
/// A twist with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::msg::Twist,

}



impl Default for TwistStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::TwistStamped::default())
  }
}

impl rosidl_runtime_rs::Message for TwistStamped {
  type RmwMsg = super::msg::rmw::TwistStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        twist: super::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(msg.twist)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        twist: super::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(&msg.twist)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      twist: super::msg::Twist::from_rmw_message(msg.twist),
    }
  }
}


// Corresponds to geometry_msgs__msg__TwistWithCovariance
/// This expresses velocity in free space with uncertainty.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistWithCovariance {

    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::msg::Twist,

    /// Row-major representation of the 6x6 covariance matrix
    /// The orientation parameters use a fixed-axis representation.
    /// In order, the parameters are:
    /// (x, y, z, rotation about X axis, rotation about Y axis, rotation about Z axis)
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub covariance: [f64; 36],

}



impl Default for TwistWithCovariance {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::TwistWithCovariance::default())
  }
}

impl rosidl_runtime_rs::Message for TwistWithCovariance {
  type RmwMsg = super::msg::rmw::TwistWithCovariance;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        twist: super::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(msg.twist)).into_owned(),
        covariance: msg.covariance,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        twist: super::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(&msg.twist)).into_owned(),
        covariance: msg.covariance,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      twist: super::msg::Twist::from_rmw_message(msg.twist),
      covariance: msg.covariance,
    }
  }
}


// Corresponds to geometry_msgs__msg__TwistWithCovarianceStamped
/// This represents an estimated twist with reference coordinate frame and timestamp.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct TwistWithCovarianceStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub twist: super::msg::TwistWithCovariance,

}



impl Default for TwistWithCovarianceStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::TwistWithCovarianceStamped::default())
  }
}

impl rosidl_runtime_rs::Message for TwistWithCovarianceStamped {
  type RmwMsg = super::msg::rmw::TwistWithCovarianceStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        twist: super::msg::TwistWithCovariance::into_rmw_message(std::borrow::Cow::Owned(msg.twist)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        twist: super::msg::TwistWithCovariance::into_rmw_message(std::borrow::Cow::Borrowed(&msg.twist)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      twist: super::msg::TwistWithCovariance::from_rmw_message(msg.twist),
    }
  }
}


// Corresponds to geometry_msgs__msg__Vector3
/// This represents a vector in free space.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
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
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Vector3::default())
  }
}

impl rosidl_runtime_rs::Message for Vector3 {
  type RmwMsg = super::msg::rmw::Vector3;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        x: msg.x,
        y: msg.y,
        z: msg.z,
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
      x: msg.x,
      y: msg.y,
      z: msg.z,
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      x: msg.x,
      y: msg.y,
      z: msg.z,
    }
  }
}


// Corresponds to geometry_msgs__msg__Vector3Stamped
/// This represents a Vector3 with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Vector3Stamped {
    /// Note that this follows vector semantics with it always anchored at the origin,
    /// so the rotational elements of a transform are the only parts applied when transforming.
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub vector: super::msg::Vector3,

}



impl Default for Vector3Stamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Vector3Stamped::default())
  }
}

impl rosidl_runtime_rs::Message for Vector3Stamped {
  type RmwMsg = super::msg::rmw::Vector3Stamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        vector: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.vector)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        vector: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.vector)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      vector: super::msg::Vector3::from_rmw_message(msg.vector),
    }
  }
}


// Corresponds to geometry_msgs__msg__VelocityStamped
/// This expresses the timestamped velocity vector of a frame 'body_frame_id' in the reference frame 'reference_frame_id' expressed from arbitrary observation frame 'header.frame_id'.
/// - If the 'body_frame_id' and 'header.frame_id' are identical, the velocity is observed and defined in the local coordinates system of the body
///   which is the usual use-case in mobile robotics and is also known as a body twist.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct VelocityStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub body_frame_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub reference_frame_id: std::string::String,


    // This member is not documented.
    #[allow(missing_docs)]
    pub velocity: super::msg::Twist,

}



impl Default for VelocityStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::VelocityStamped::default())
  }
}

impl rosidl_runtime_rs::Message for VelocityStamped {
  type RmwMsg = super::msg::rmw::VelocityStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        body_frame_id: msg.body_frame_id.as_str().into(),
        reference_frame_id: msg.reference_frame_id.as_str().into(),
        velocity: super::msg::Twist::into_rmw_message(std::borrow::Cow::Owned(msg.velocity)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        body_frame_id: msg.body_frame_id.as_str().into(),
        reference_frame_id: msg.reference_frame_id.as_str().into(),
        velocity: super::msg::Twist::into_rmw_message(std::borrow::Cow::Borrowed(&msg.velocity)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      body_frame_id: msg.body_frame_id.to_string(),
      reference_frame_id: msg.reference_frame_id.to_string(),
      velocity: super::msg::Twist::from_rmw_message(msg.velocity),
    }
  }
}


// Corresponds to geometry_msgs__msg__Wrench
/// This represents force in free space, separated into its linear and angular parts.

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Wrench {

    // This member is not documented.
    #[allow(missing_docs)]
    pub force: super::msg::Vector3,


    // This member is not documented.
    #[allow(missing_docs)]
    pub torque: super::msg::Vector3,

}



impl Default for Wrench {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::Wrench::default())
  }
}

impl rosidl_runtime_rs::Message for Wrench {
  type RmwMsg = super::msg::rmw::Wrench;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        force: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.force)).into_owned(),
        torque: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Owned(msg.torque)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        force: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.force)).into_owned(),
        torque: super::msg::Vector3::into_rmw_message(std::borrow::Cow::Borrowed(&msg.torque)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      force: super::msg::Vector3::from_rmw_message(msg.force),
      torque: super::msg::Vector3::from_rmw_message(msg.torque),
    }
  }
}


// Corresponds to geometry_msgs__msg__WrenchStamped
/// A wrench with reference coordinate frame and timestamp

#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct WrenchStamped {

    // This member is not documented.
    #[allow(missing_docs)]
    pub header: std_msgs::msg::Header,


    // This member is not documented.
    #[allow(missing_docs)]
    pub wrench: super::msg::Wrench,

}



impl Default for WrenchStamped {
  fn default() -> Self {
    <Self as rosidl_runtime_rs::Message>::from_rmw_message(super::msg::rmw::WrenchStamped::default())
  }
}

impl rosidl_runtime_rs::Message for WrenchStamped {
  type RmwMsg = super::msg::rmw::WrenchStamped;

  fn into_rmw_message(msg_cow: std::borrow::Cow<'_, Self>) -> std::borrow::Cow<'_, Self::RmwMsg> {
    match msg_cow {
      std::borrow::Cow::Owned(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Owned(msg.header)).into_owned(),
        wrench: super::msg::Wrench::into_rmw_message(std::borrow::Cow::Owned(msg.wrench)).into_owned(),
      }),
      std::borrow::Cow::Borrowed(msg) => std::borrow::Cow::Owned(Self::RmwMsg {
        header: std_msgs::msg::Header::into_rmw_message(std::borrow::Cow::Borrowed(&msg.header)).into_owned(),
        wrench: super::msg::Wrench::into_rmw_message(std::borrow::Cow::Borrowed(&msg.wrench)).into_owned(),
      })
    }
  }

  fn from_rmw_message(msg: Self::RmwMsg) -> Self {
    Self {
      header: std_msgs::msg::Header::from_rmw_message(msg.header),
      wrench: super::msg::Wrench::from_rmw_message(msg.wrench),
    }
  }
}


