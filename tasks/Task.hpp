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

/** EDS Library **/
#include <eds/EDS.h>

/** Yaml **/
#include <yaml-cpp/yaml.h>

/** std **/
#include <memory> //shared_pointer

namespace eds{


    /*! \class Task
     * \brief The task context provides and requires services. It uses an ExecutionEngine to perform its functions.
     * Essential interfaces are operations, data flow ports and properties. These interfaces have been defined using the oroGen specification.
     * In order to modify the interfaces you should (re)use oroGen and rely on the associated workflow.
     * Camera calibration file
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

        /** Calibration **/
        ::eds::calib::DualCamera cam_calib;
        std::shared_ptr<::eds::calib::Camera> cam0;
        std::shared_ptr<::eds::calib::Camera> cam1;
        std::shared_ptr<::eds::calib::Camera> newcam;

        /** Indexes **/
        uint64_t ef_idx, frame_idx;
        int64_t kf_idx;

        /** Initialized flag **/
        bool initialized, first_track;

        /** Is Lost flag **/
        bool is_lost;

        float rescale_factor;
        std::vector<float> all_scale;

        /** Image frame in opencv format **/
        cv::Mat img_frame;

        /** Image frame in opencv format splitted in channels **/
        cv::Mat img_rgb[3];

        /** Buffer of events **/
        std::vector<::base::samples::Event> events;

        /** Local Depth map **/
        std::shared_ptr<::eds::mapping::IDepthMap2d> depthmap;

        /** All Frames and Keyframes History **/
        std::vector<dso::FrameShell*> all_frame_history;
        std::vector<dso::FrameShell*> all_keyframes_history;

        /** Keyframes window **/
        std::vector<dso::FrameHessian*> frame_hessians;

        /** DSO Calibration **/
        std::shared_ptr<dso::CalibHessian> calib;

        /** DSO Undistort parameters **/
        std::shared_ptr<::dso::Undistort> undistort;

        /**  Initializer **/
        std::shared_ptr<::dso::CoarseInitializer> initializer;

        /** Event-to-Image tracker (EDS)**/
        std::shared_ptr<::eds::tracking::Tracker> event_tracker;
        std::shared_ptr<::eds::tracking::KeyFrame> key_frame;
        std::shared_ptr<::eds::tracking::EventFrame> event_frame;

        /** Image-to-Image Tracker (DSO) **/
        std::shared_ptr<::dso::CoarseTracker> image_tracker;
        dso::Vec5 last_coarse_RMSE;

        /** Mapping (Point selection strategy)**/
        float* selection_map;
        std::shared_ptr<::dso::PixelSelector> pixel_selector;
        std::shared_ptr<::dso::CoarseDistanceMap> coarse_distance_map;

        /** Bundle Adjustment using DSO energy functional **/
        float currentMinActDist;
        std::vector<dso::PointFrameResidual*> active_residuals;
        ::dso::IndexThreadReduce<dso::Vec10> thread_reduce;
        std::shared_ptr<::dso::EnergyFunctional> bundles;
        std::vector<float> all_res_vec;

        /** KeyFrame camera pose w.r.t World **/
        base::samples::RigidBodyState pose_w_kf;

        /** EventFrame camera pose w.r.t Keyframe **/
        base::samples::RigidBodyState pose_kf_ef;

        /** EventFrame camera pose w.r.t World **/
        base::samples::RigidBodyState pose_w_ef;

        /** Globa Map **/
        std::map<int, base::samples::Pointcloud> global_map;

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

    protected:

        template<typename T> inline void deleteOut(std::vector<T*> &v, const int i)
        {
            delete v[i];
            v[i] = v.back();
            v.pop_back();
        }

        template<typename T> inline void deleteOutOrder(std::vector<T*> &v, const T* element)
        {
            int i=-1;
            for(unsigned int k=0; k<v.size();k++)
            {
                if(v[k] == element)
                {
                    i=k;
                    break;
                }
            }
            assert(i!=-1);

            for(unsigned int k=i+1; k<v.size();k++)
                v[k-1] = v[k];
            v.pop_back();

            delete element;
        }

        /** Task data loader config **/
        ::eds::DataLoaderConfig readDataLoaderConfig(YAML::Node config);

        /** Get Input Data **/
        dso::ImageAndExposure* getImageAndExposure(const ::base::samples::frame::Frame &frame);

        /** Initializer **/
        dso::SE3 initialize(dso::ImageAndExposure* image, int id, const int &snapped_threshold);

        /** Events to Image Tracker. This is the EDS tracker**/
        bool eventsToImageAlignment(const std::vector<::base::samples::Event> &events_array, ::base::Transform3d &T_kf_ef);

        /** Image to Image Tracker. DSO-based**/
        void setPrecalcValues();
        void track(dso::ImageAndExposure* image, int id);
        dso::Vec4 trackNewFrame(dso::FrameHessian* fh, const dso::SE3 &event_trans);
        dso::Vec4 recoveryTracking(dso::FrameHessian* fh);
        void makeNonKeyFrame(dso::FrameHessian* fh);
        void makeKeyFrame(dso::FrameHessian* fh);
        void traceNewPoints(dso::FrameHessian* fh);
        void makeNewTraces(dso::FrameHessian* newFrame, ::eds::mapping::IDepthMap2d *depthmap);

        /** Points Optimization and Backend **/
        void activatePointsMT();
        void activatePointsMT_Reductor(std::vector<dso::PointHessian*>* optimized, std::vector<dso::ImmaturePoint*>* toOptimize, int min, int max, dso::Vec10* stats, int tid);
        dso::PointHessian* optimizeImmaturePoint(dso::ImmaturePoint* point, int minObs, dso::ImmaturePointTemporaryResidual* residuals);
        void removeOutliers();
        void flagPointsForRemoval();
        std::vector<dso::VecX> getNullspaces(
            std::vector<dso::VecX> &nullspaces_pose, std::vector<dso::VecX> &nullspaces_scale,
            std::vector<dso::VecX> &nullspaces_affA, std::vector<dso::VecX> &nullspaces_affB);
        void solveSystem(int iteration, double lambda);
        bool doStepFromBackup(float stepfacC,float stepfacT,float stepfacR,float stepfacA,float stepfacD);
        void backupState(bool backupLastStep);
        void loadSateBackup();
        double calcMEnergy();
        double calcLEnergy();
        void applyRes_Reductor(bool copyJacobians, int min, int max, dso::Vec10* stats, int tid);
        void linearizeAll_Reductor(bool fixLinearization, std::vector<dso::PointFrameResidual*>* toRemove, int min, int max, dso::Vec10* stats, int tid);
        void setNewFrameEnergyTH();
        dso::Vec3 linearizeAll(bool fixLinearization);
        float optimize(int mnumOptIts);
        void printOptRes(const dso::Vec3 &res, double resL, double resM, double resPrior, double LExact, float a, float b);

        /** Marginalization **/
        void flagFramesForMarginalization(dso::FrameHessian* newFH);
        void marginalizeFrame(dso::FrameHessian* frame);

        /** Get local points **/
        std::vector<base::Point> getPoints(const ::base::Transform3d &T_kf_w = ::base::Transform3d::Identity(), const bool &single_point = true);
        std::vector<cv::Point2d> getImmatureCoords(const dso::FrameHessian *fh);

        /** Two version of get Maps **/
        base::samples::Pointcloud getMap(const bool &single_point, const bool &color=true);
        void getMap(std::map<int, base::samples::Pointcloud> &global_map, const bool &single_point, const bool &color=true);

        /** Output poses **/
        void outputPose(const dso::FrameShell *current_frame, const base::Time &timestamp);
        void outputPoseKF(const dso::FrameHessian *current_kf, const base::Time &timestamp);
        void outputPoseKFs(const base::Time &timestamp);

        /** Output statistics info **/
        void outputTrackerInfo(const ::base::Time &timestamp);

        /** Output visuals information **/
        void outputEventFrameViz(const std::shared_ptr<::eds::tracking::EventFrame> &event_frame);
        void outputGenerativeModelFrameViz(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputInvDepthAndLocalMap(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputOpticalFlowFrameViz(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputGradientsFrameViz(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp);
        void outputDepthMap(const std::shared_ptr<::eds::mapping::IDepthMap2d> &depthmap, const ::base::Time &timestamp);
        void outputLocalMap(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const std::vector<double> &idp, const std::vector<double> &model={});
        void outputKeyFrameMosaicViz(const std::vector<dso::FrameHessian*> &frame_hessians, const ::base::Time &timestamp);
        void outputImmaturePtsFrameViz(const dso::FrameHessian *input, const ::base::Time &timestamp);
        void outputGlobalMap();

        /** Print a stamped trajectory txt file **/
        void printResult(std::string file);
    };
}

#endif

