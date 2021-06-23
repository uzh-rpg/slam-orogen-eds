
Event-aided Direct Sparse Odometry (EDS)
=============
Direct visual odometry approach using events and frames (CVPR 2022)
This is the Rock component (equivalent to a ROS node) which uses the
C++ [EDS library](https://github.com/uzh-rpg/slam-eds) to create a Task. 

License
-------
See LICENSE file


Installation
------------
The easiest way to build and install this component is to use Rock's build system.
See [this page](http://rock-robotics.org/documentation/installation.html)
on how to install Rock. You can use the [EDS builconf](https://github.com/uzh-rpg/eds-buildconf)
which tells you how to install and use the component from scratch. There is also the opction to directly
use this component in a Docker container.

This component is a basic Task class which wraps the EDS library in an RTT component style. Therfore, it depends
on RTT communication layer in which Rock is built on top. It should be straight forward to remove this dependency and use
the Task class as a single main program. 

Dependencies
-----------------
Dependencies are listed in the manifest file, those are:

* [base/cmake](https://github.com/rock-core/base-cmake): the CMake pure function to build this library
* [slam/eds](https://github.com/uzh-rpg/slam-eds): the EDS library
* [yaml-cpp](https://github.com/jbeder/yaml-cpp): YAML config parser for the configuration parameters
* [perception/frame_helper](https://github.com/rock-core/perception-frame_helper): Frame Image converter between Opencv and Rock types
* [base/orogen/types](https://github.com/rock-core/base-orogen-types): Package to expose the Rock (Robot Construction Kit) base types
* [drivers/orogen/aggregrator](https://github.com/rock-core/drivers-orogen-aggregator): This is a data stream aligner. It brings the callback functionaly to the Task class.


OROGEN file
-----------------
Tasks are defined in Orogen which is a subset of Ruby language. The orogen file (eds.orogen) defines
all the Task functionalities such as properties, input & output ports, triggering mechanism, etc..
This files is parsed by Rock in order to defined the temporary content in the .orogen folder.
You can call the orogen parser but just typing:

```console
docker@eds:~S orogen --corba eds.orogen
```

Task Standard Layout
--------------------
This directory structure follows some simple rules, to allow for generic build
processes and simplify reuse of this project. Following these rules ensures that
the Rock CMake macros automatically handle the project's build process and
install setup properly.

### EDS Folder Structure

| directory         |       purpose                                                        |
| ----------------- | ------------------------------------------------------               |
| task/             | Contains the Task.cpp and Task.hpp of the [component](https://www.rock-robotics.org/documentation/orogen/task_interface.html)                  |
| .orogen/          | Contains the Task parent classes which interface with [RTT](https://www.orocos.org/rtt/)  <br> This is a temporary content folder created from eds.orogen <br> This is the folder to remove (to make Task not inherit) in order to make this an independent main program |
| build/ *          | The target directory for the build process, temporary content        |
| [edsTypes.hpp](edsTypes.hpp)      | Specific data types for the Task. These are normally types which are not available in [base-types](https://www.rock-robotics.org/documentation/orogen/task_interface.html). These are normally cofiguration struct or specific output port data types. |
| [eds.orogen](eds.orogen)        | This is the task definition file                 |
