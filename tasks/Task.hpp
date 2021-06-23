/*
* This file is part of the EDS: Event-aided Direct Sparse Odometry
* (https://rpg.ifi.uzh.ch/eds.html)
*
* Copyright (c) 2022 Javier Hidalgo-Carrio, Robotics and Perception
* Group (RPG) University of Zurich.
*
* EDS is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* EDS is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef EDS_TASK_TASK_HPP
#define EDS_TASK_TASK_HPP

#include "eds/TaskBase.hpp"

/** EDS library **/
#include "eds/edsTypes.hpp"
#include "eds/tracking/EventFrame.hpp"
#include "eds/tracking/KeyFrame.hpp"
#include "eds/tracking/Tracker.hpp"
#include "eds/mapping/GlobalMap.hpp"
#include "eds/bundles/BundleAdjustment.hpp"

/** Yaml **/
#include <yaml-cpp/yaml.h>


namespace eds{


    /*! \class Task
     * \brief The task context provides and requires services. It uses an ExecutionEngine to perform its functions.
     * Essential interfaces are operations, data flow ports and properties. These interfaces have been defined using the oroGen specification.
     * In order to modify the interfaces you should (re)use oroGen and rely on the associated workflow.
     * Yaml file for EDS configuration
     * \details
     * The name of a TaskContext is primarily defined via:
     \verbatim
     deployment 'deployment_name'
         task('custom_task_name','eds::Task')
     end
     \endverbatim
     *  It can be dynamically adapted when the deployment is called with a prefix argument.
     */
    class Task : public TaskBase
    {
	friend class TaskBase;

    protected:

        /** Configuration **/
        EDSConfiguration eds_config;
        ::eds::calib::CameraInfo frame_cam_info, event_cam_info;
        ::base::Transform3d T_cam_imu, T_init;

        /** Counters **/
        uint64_t ef_idx, kf_idx, init_frames;

        /** Initial Scale **/
        double init_scale;

        /** Tasks flags **/
        bool got_first_event_frame;
        bool create_kf;

        /** Buffer of events **/
        std::vector<::base::samples::Event> events;

        /** Buffer of inertial measurements **/
        std::vector<::base::samples::IMUSensors> imus;

        /** Local Depth map **/
        ::eds::mapping::IDepthMap2d depthmap;

        /** Image frame in opencv format **/
        cv::Mat img_frame;

        /** Task member variables **/
        std::shared_ptr<::eds::tracking::Tracker> tracker;
        std::shared_ptr<::eds::tracking::KeyFrame> kf;
        std::shared_ptr<::eds::tracking::EventFrame> ef;
        std::shared_ptr<::eds::mapping::GlobalMap> global_map;
        std::shared_ptr<::eds::bundles::BundleAdjustment> bundles;

        /** Monitoring information **/
        eds::TrackerInfo tracker_info;
        eds::PBAInfo bundles_info;

        /** KeyFrame camera pose w.r.t World **/
        base::samples::RigidBodyState pose_w_kf;

        /** EventFrame camera pose w.r.t Keyframe **/
        base::samples::RigidBodyState pose_kf_ef;

        /** EventFrame camera pose w.r.t World **/
        base::samples::RigidBodyState pose_w_ef;

        /** Ground Truth camera pose w.r.t World **/
        base::samples::RigidBodyState pose_w_gt;

    protected:

        virtual void eventsCallback(const base::Time &ts, const ::base::samples::EventArray &events_sample);

        virtual void frameCallback(const base::Time &ts, const ::RTT::extras::ReadOnlyPointer< ::base::samples::frame::Frame > &frame_sample);

        virtual void imuCallback(const base::Time &ts, const ::base::samples::IMUSensors &imu_sample);

        virtual void groundtruthCallback(const base::Time &ts, const ::base::samples::RigidBodyState &groundtruth_sample);

    public:
        /** TaskContext constructor for Task
         * \param name Name of the task. This name needs to be unique to make it identifiable via nameservices.
         * \param initial_state The initial TaskState of the TaskContext. Default is Stopped state.
         */
        Task(std::string const& name = "eds::Task");

        /** Default deconstructor of Task
         */
	    ~Task();

        /** This hook is called by Orocos when the state machine transitions
         * from PreOperational to Stopped. If it returns false, then the
         * component will stay in PreOperational. Otherwise, it goes into
         * Stopped.
         *
         * It is meaningful only if the #needs_configuration has been specified
         * in the task context definition with (for example):
         \verbatim
         task_context "TaskName" do
           needs_configuration
           ...
         end
         \endverbatim
         */
        bool configureHook();

        /** This hook is called by Orocos when the state machine transitions
         * from Stopped to Running. If it returns false, then the component will
         * stay in Stopped. Otherwise, it goes into Running and updateHook()
         * will be called.
         */
        bool startHook();

        /** This hook is called by Orocos when the component is in the Running
         * state, at each activity step. Here, the activity gives the "ticks"
         * when the hook should be called.
         *
         * The error(), exception() and fatal() calls, when called in this hook,
         * allow to get into the associated RunTimeError, Exception and
         * FatalError states.
         *
         * In the first case, updateHook() is still called, and recover() allows
         * you to go back into the Running state.  In the second case, the
         * errorHook() will be called instead of updateHook(). In Exception, the
         * component is stopped and recover() needs to be called before starting
         * it again. Finally, FatalError cannot be recovered.
         */
        void updateHook();

        /** This hook is called by Orocos when the component is in the
         * RunTimeError state, at each activity step. See the discussion in
         * updateHook() about triggering options.
         *
         * Call recover() to go back in the Runtime state.
         */
        void errorHook();

        /** This hook is called by Orocos when the state machine transitions
         * from Running to Stopped after stop() has been called.
         */
        void stopHook();

        /** This hook is called by Orocos when the state machine transitions
         * from Stopped to PreOperational, requiring the call to configureHook()
         * before calling start() again.
         */
        void cleanupHook();

    private:

        ::eds::DataLoaderConfig readDataLoaderConfig(YAML::Node config);
        ::eds::calib::CameraInfo readCameraCalib(YAML::Node cam_calib);
        ::base::Transform3d readInitPose(YAML::Node cam_calib);

        cv::Mat depthmapUndistort(const ::eds::calib::CameraInfo &cam_info, const cv::Mat &img);

        void imuMeanUnitVector(std::vector<::base::samples::IMUSensors> &data,
                            const ::base::Transform3d &T_cam_imu, const Eigen::Matrix3d &R_w_cam,
                            ::base::Vector3d &gyros, ::base::Vector3d &acc);

        bool eventsToImageAlignment(const std::vector<::base::samples::Event> &events_array, ::base::Transform3d &T_kf_ef);

        bool bootstrapping(const ::base::Time &timestamp);

        bool isInitialized();
        bool createNewKeyFrame(const ::base::Transform3d &T_kf_ef);
        void outputEventFrameViz();

        void outputTrackerInfo(const ::base::Time &timestamp);
        void outputBundlesInfo(const ::base::Time &timestamp);

        void outputPoseKFs();
        void outputKeyFrameViz();
        void outputGlobalMap();

        void outputOpticalFlowFrameViz(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputGradients(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputModel(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputInvDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const base::Transform3d &T_kf_ef, const ::base::Time &timestamp);
        void outputSigmaInvDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputConvergenceDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputGlobalMapMosaic(const std::shared_ptr<::eds::mapping::GlobalMap> &global_map, const ::base::Time &timestamp);
    };
}

#endif

