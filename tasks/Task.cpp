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

#include <frame_helper/FrameHelper.h>
#include <opencv2/imgproc.hpp>

#include "Task.hpp"

using namespace eds;

#define DEBUG_PRINTS 1
#define GRAVITY 9.81

Task::Task(std::string const& name)
    : TaskBase(name)
{
    this->kf.reset();
    this->ef.reset();
    this->tracker.reset();
    this->global_map.reset();
}

Task::~Task()
{
}

void Task::eventsCallback(const base::Time &ts, const ::base::samples::EventArray &events_sample)
{
    //#ifdef DEBUG_PRINTS
    //std::cout<<"** [EDS_TASK EVENTS] Received "<<events_sample.events.size()<<" events at["<<events_sample.time.toSeconds()<<"] **\n";
    //#endif

    /** Insert Events into the buffer **/
    this->events.insert(this->events.end(), events_sample.events.begin(), events_sample.events.end());

    /** Create an event frame **/
    if (this->events.size() >= this->eds_config.data_loader.num_events)
    {
        #ifdef DEBUG_PRINTS
        ::base::Time first_ts = this->events[0].ts;
        ::base::Time last_ts = this->events[this->eds_config.data_loader.num_events-1].ts;
        std::cout<<"** [EDS_TASK EVENTS] Processing events from["<<first_ts.toSeconds()<<"] to["<<last_ts.toSeconds()<<"] size:"<<this->events.size()<<std::endl;
        #endif

        /** Move data (no copy) **/
        std::vector<::base::samples::Event> ef_events;
        std::move(this->events.begin(), this->events.begin()+this->eds_config.data_loader.num_events, std::back_inserter(ef_events)); 

        /** Clean events in the original buffer depending in the overlap percentage.
         * std move only moves (it does not delete) **/
        int next_element = (1.0 - this->eds_config.data_loader.overlap)*this->eds_config.data_loader.num_events;
        this->events.erase(this->events.begin(), this->events.begin()+next_element);
        #ifdef DEBUG_PRINTS
        std::cout<<"** [EDS_TASK EVENTS] ef_events at["<<ef_events[0].ts.toMicroseconds()<<"] size:"<<ef_events.size()<<std::endl;
        std::cout<<"** [EDS_TASK EVENTS] overlap ["<< this->eds_config.data_loader.overlap*100.0 <<"] this->events at["<<this->events[0].ts.toMicroseconds()<<"] size:"<<this->events.size()<<std::endl;
        #endif

        /** Create the Event Frame **/
        this->ef = std::make_shared<eds::tracking::EventFrame>(this->ef_idx, ef_events, this->event_cam_info,
                                                            this->eds_config.tracker.options.max_num_iterations.size(),
                                                            base::Affine3d::Identity(),
                                                            cv::Size(this->event_cam_info.out_width, this->event_cam_info.out_height));

        /** Got first image **/
        if (this->got_first_event_frame == false)
            this->got_first_event_frame = true;

        /** Set to initializing if this is the first eventframe (NOTE: comment state setting for GT Map testing) **/
        if (this->ef_idx == 0) state(INITIALIZING);

        /** Increment EF INDEX **/
        this->ef_idx++;

        /** TRACKER OPTIMIZATION NOTE: uncoment kf_idx for GT Map**/
        if (state() == RUNNING /*and this->kf_idx>0*/)
        {
            /** Event to Image alignment T_kf_ef delta pose **/
            ::base::Transform3d T_kf_ef; this->eventsToImageAlignment(ef_events, T_kf_ef);

            /** Set the EventFrame pose: T_w_ef with the result from alignment**/
            this->ef->setPose(this->pose_w_kf.getTransform()*T_kf_ef);

            /** Update the output port: T_kf_ef **/
            this->pose_kf_ef.time = this->ef->time;
            this->pose_kf_ef.setTransform(T_kf_ef);
            #ifdef DEBUG_PRINTS
            std::cout<<"** [EDS_TASK EVENTS] Wrote pose_kf_ef:\n"<<this->pose_kf_ef.getTransform().matrix()<<std::endl;
            #endif

            /** Write output port: T_w_ef **/
            this->pose_w_ef.time = this->ef->time;
            this->pose_w_ef.setTransform(this->ef->getPose());
            this->pose_w_ef.velocity = this->pose_w_ef.getTransform() * this->tracker->linearVelocity();
            this->pose_w_ef.angular_velocity = this->pose_w_ef.getTransform() * this->tracker->angularVelocity();
            _pose_w_ef.write(this->pose_w_ef);
            #ifdef DEBUG_PRINTS
            std::cout<<"** [EDS_TASK EVENTS] Wrote pose_w_ef:\n"<<this->pose_w_ef.getTransform().matrix()<<std::endl;
            #endif

            state(LOGGING);
            /** Write the event frame **/
            this->outputEventFrameViz();

            /** Output Model **/
            this->outputModel(this->kf, this->ef->time);

            /** Output Gradients **/
            this->outputGradients(this->kf, this->ef->time);

            /** Ouput the optical flow **/
            this->outputOpticalFlowFrameViz(this->kf, this->ef->time);

            /** Output Inverse depth **/
            this->outputInvDepth(this->kf, T_kf_ef, this->ef->time);

            /** Output Inverse sigma depth **/
            this->outputSigmaInvDepth(this->kf, this->ef->time);

            /** Output Inverse depth convergence **/
            this->outputConvergenceDepth(this->kf, this->ef->time);

            /** Tracker infos **/
            this->outputTrackerInfo(this->ef->time);

            state(RUNNING);

        }
    }

}

void Task::frameCallback(const base::Time &ts, const ::RTT::extras::ReadOnlyPointer< ::base::samples::frame::Frame > &frame_sample)
{
    #ifdef DEBUG_PRINTS
    std::cout<<"** [EDS_TASK FRAME] Received Frame at ["<<frame_sample->time.toSeconds()<<"] Create NEW KF ["<<(this->create_kf?"True":"False")<<"] WITH ID["<<this->kf_idx<<"]**\n";
    #endif

    /** Read depth map (if any, only debug) **/
    RTT::extras::ReadOnlyPointer<::base::samples::DistanceImage> depthmap_ptr;
    //RTT::extras::ReadOnlyPointer<::base::samples::frame::Frame> depthmap_ptr;
    if (_depthmap.read(depthmap_ptr, false) == RTT::NewData)
    {
        #ifdef DEBUG_PRINTS
        std::cout<<"** [EDS_TASK FRAME] Received Depthmap at ["<<depthmap_ptr->time.toSeconds()<<"] **\n";
        #endif
        //this->depthmap.fromDistanceImage(*depthmap_ptr); //NOTE: uncomment this line for GT Map testing
        //NOTE: uncomment the lines bellow in case you generate depthmaps in image format
        //cv::Mat img = this->depthmapUndistort(this->frame_cam_info,  frame_helper::FrameHelper::convertToCvMat(*depthmap_ptr));
        //this->depthmap.fromDepthmapImage(img, this->frame_cam_info.intrinsics,
        //                    this->eds_config.mapping.min_depth, this->eds_config.mapping.max_depth);
    }

    if (!this->got_first_event_frame)
    {
        #ifdef DEBUG_PRINTS
        std::cout<<"** [EDS_TASK FRAME] Waiting for the first event frame: Events array size: "<<this->events.size()<<std::endl;
        #endif
        return;
    }

    /** Get the image frame in color **/
    this->img_frame = frame_helper::FrameHelper::convertToCvMat(*frame_sample);
    if (this->frame_cam_info.flip) cv::flip(this->img_frame, this->img_frame, 1);

    if (state() == INITIALIZING)
    {
        /** First Keyframe **/
        if (this->kf_idx == 0)
        {
            /** The pose for the next KF: T_w_kf(k) **/
            ::base::Transform3d T_w_kf = this->pose_w_kf.getTransform() * this->pose_kf_ef.getTransform();

            /** Reset the Keyframe to Eventframe transformation **/
            this->pose_kf_ef.setTransform(::base::Transform3d::Identity());

            /** Create the Keyframe **/
            this->kf = std::make_shared<eds::tracking::KeyFrame>(this->kf_idx, frame_sample->time, this->img_frame,
                            this->depthmap, this->frame_cam_info, this->eds_config.mapping, this->eds_config.tracker.percent_points,
                            T_w_kf, cv::Size(this->frame_cam_info.out_width, this->frame_cam_info.out_height));

            /** Increment KF INDEX **/
            this->kf_idx++;
        }

        /** Bootstrapping **/
        if (this->bootstrapping(frame_sample->time))
        {
            /* Update the pose_w_kf **/
            this->pose_w_kf.time = this->kf->time;
            this->pose_w_kf.setTransform(this->kf->getPose());

            /******  OUPUT INFO  *******/
            state(LOGGING);
            this->outputPoseKFs();
            this->outputGlobalMap();
            this->outputKeyFrameViz();

            /** Set state to running **/
            state(RUNNING);
        }
    }
    if (state() == RUNNING and this->create_kf)
    {
        /** NOTE uncoment this line when want to test with GT maps **/
        //if (this->depthmap.empty()) return;

        /** The pose for the next KF: T_w_kf(k) **/
        ::base::Transform3d T_w_kf = this->pose_w_kf.getTransform() * this->pose_kf_ef.getTransform();

        /** Reset the Keyframe to Eventframe transformation **/
        this->pose_kf_ef.setTransform(::base::Transform3d::Identity());

        /** Get the (initial) global map in KF without optimization (NOTE: comment this when GT maps available for testing)  **/
        this->global_map->getIDepthMap(this->kf->idx, this->depthmap, true, false);

        /** Create the Keyframe **/
        this->kf = std::make_shared<eds::tracking::KeyFrame>(this->kf_idx, frame_sample->time, this->img_frame,
                        this->depthmap, this->frame_cam_info, this->eds_config.mapping, this->eds_config.tracker.percent_points,
                        T_w_kf, cv::Size(this->frame_cam_info.out_width, this->frame_cam_info.out_height));

        /** Points selection refinement **/
        this->kf->pointsRefinement(this->ef->frame[0]);

        /** Insert the KF in the global map **/
        ::eds::mapping::POINTS_SELECT_METHOD method = eds::mapping::POINTS_SELECT_METHOD::NONE;
        global_map->insert(this->kf, method);

        /**  Photometric Bundle Adjustement (optimize) NOTE: comment when no required**/
        state(OPTIMIZING);
        //uint64_t kf_to_marginalize = this->global_map->prev_last_kf_id; //NOTE uncomment this for not optimizing but still keeping the windows size
        uint64_t kf_to_marginalize;
        this->bundles->optimize(this->global_map, kf_to_marginalize, eds::bundles::MAD, true);
        //this->global_map->cleanMap(1);
        state(RUNNING);

        /** Get the transformation of the optimized KF: T_w_kf**/
        this->pose_w_kf.setTransform(global_map->getKFTransform(this->kf->idx));
        this->pose_w_kf.time = this->kf->time;

        /** Update the KF pose with the optimized transformation **/
        this->kf->setPose(this->pose_w_kf.getTransform());

        /** Get the Global Map in the optimized KF pose (NOTE: comment this when GT maps test) **/
        this->global_map->getIDepthMap(this->kf->idx, this->depthmap, true, false);

        /** Update the KF depthmap with the optimized one **/
        this->kf->setDepthMap(this->depthmap, this->eds_config.mapping, {1.0, 1.0});

        /** Reset the tracker with the new keyframe **/
        this->tracker->reset(this->kf, Eigen::Vector3d::Zero(), Eigen::Quaterniond::Identity());

        /** Increment KF INDEX **/
        this->kf_idx++;

        /** Reset creation of KF **/
        this->create_kf = false;

        /***** OUTPUT PORTS *********/
        state(LOGGING);
        this->outputPoseKFs();
        this->outputGlobalMap();
        this->outputKeyFrameViz();
        this->outputBundlesInfo(this->kf->time);
        state(RUNNING);
        /***** END OUTPUT PORTS *****/

        /** Marginalize KF when necessary **/
        if (kf_to_marginalize != ::eds::mapping::NONE_KF)
        {
            this->global_map->removeKeyFrame(kf_to_marginalize);
        }
    }
}

void Task::imuCallback(const base::Time &ts, const ::base::samples::IMUSensors &imu_sample)
{
    /** Buffer of inertial measurements window **/
    //this->imus.push_back(imu_sample);
}

void Task::groundtruthCallback(const base::Time &ts, const ::base::samples::RigidBodyState &groundtruth_sample)
{
    this->pose_w_gt = groundtruth_sample;
    this->_pose_w_gt.write(groundtruth_sample);
}

/// The following lines are template definitions for the various state machine
// hooks defined by Orocos::RTT. See Task.hpp for more detailed
// documentation about them.

bool Task::configureHook()
{
    if (! TaskBase::configureHook())
        return false;

    /** Read the Yaml configuration file **/
    YAML::Node config = YAML::LoadFile(_config_file.get());
    eds_config.data_loader = this->readDataLoaderConfig(config["data_loader"]);
    eds_config.tracker = ::eds::tracking::readTrackingConfig(config["tracker"]);
    eds_config.mapping = ::eds::mapping::readMappingConfig(config["mapping"]);
    eds_config.bundles = ::eds::bundles::readBundlesConfig(config["bundles"]);
    eds_config.recorder = ::eds::recorder::readRecorderConfig(config["recorder"]);

    /** Read the camera calibration **/
    YAML::Node node_info = YAML::LoadFile(_calib_file.get());
    this->frame_cam_info = this->readCameraCalib(node_info["cam0"]);
    if (node_info["cam1"])
        this->event_cam_info = this->readCameraCalib(node_info["cam1"]);
    else
        this->event_cam_info = this->frame_cam_info;

    auto print = [](std::vector<double> &vector)
    {
        for (auto it: vector) {std::cout<<it<<", ";}
    };
    std::cout<<"* [EDS_TASK CONFIG] Frame intrinsics [";print(this->frame_cam_info.intrinsics); std::cout<<"]"<<std::endl;
    std::cout<<"* [EDS_TASK CONFIG] Events intrinsics [";print(this->event_cam_info.intrinsics); std::cout<<"]"<<std::endl;

    /** T_cam to imu (normaly event camera frame)**/
    Eigen::Matrix<double, 4, 4, Eigen::RowMajor> T (this->event_cam_info.T_cam_imu.data());
    this->T_cam_imu.matrix() = T;

    /** Initial scale is 1.0 **/
    this->init_scale = 1.0;

    return true;
}
bool Task::startHook()
{
    if (! TaskBase::startHook())
        return false;

    /** Reset counters **/
    this->ef_idx = this->kf_idx = 0;
    this->init_frames = 0;

    /** Set to false **/
    this->got_first_event_frame = false;
    this->create_kf = true;
 
    /** Calib file **/
    YAML::Node node_info = YAML::LoadFile(_calib_file.get());
    this->T_init = this->readInitPose(node_info);

    /** Initialize T_w_kf **/
    this->pose_w_kf.sourceFrame = "kf";
    this->pose_w_kf.targetFrame = "world";
    this->pose_w_kf.setTransform(this->T_init);

    /** Initialize T_kf_ef **/
    this->pose_kf_ef.sourceFrame = "ef";
    this->pose_kf_ef.targetFrame = "kf";
    this->pose_kf_ef.setTransform(::base::Transform3d::Identity());

    /** Initialize T_w_ef **/
    this->pose_w_ef.sourceFrame = "ef";
    this->pose_w_ef.targetFrame = "world";
    this->pose_w_ef.setTransform(this->T_init);

    /** Initialize T_w_gt **/
    this->pose_w_gt.sourceFrame = "gt";
    this->pose_w_ef.targetFrame = "world";
    this->pose_w_ef.position = ::base::NaN<double>() * Eigen::Vector3d::Ones();
    this->pose_w_ef.orientation = base::Orientation(::base::NaN<double>(), ::base::NaN<double>(),
                                    ::base::NaN<double>(), ::base::NaN<double>());

    /* Create the tracker **/
    this->tracker = std::make_shared<::eds::tracking::Tracker> (this->eds_config.tracker);

    /* Create the Global Map (use intrinsic from the frame camera) **/
    this->global_map = std::make_shared<eds::mapping::GlobalMap> (this->eds_config.mapping, this->frame_cam_info,
                                        this->eds_config.bundles, this->eds_config.tracker.percent_points);


    /** Create the Bundle Adjustement **/
    this->bundles = std::make_shared<eds::bundles::BundleAdjustment>(this->eds_config.bundles,
                        this->eds_config.mapping.min_depth, this->eds_config.mapping.max_depth);

    return true;
}

void Task::updateHook()
{
    TaskBase::updateHook();
}

void Task::errorHook()
{
    TaskBase::errorHook();
}

void Task::stopHook()
{
    TaskBase::stopHook();

    /** Clean the pointers **/
    this->kf.reset();
    this->ef.reset();
    this->tracker.reset();
    this->global_map.reset();
}

void Task::cleanupHook()
{
    TaskBase::cleanupHook();
}

::eds::DataLoaderConfig Task::readDataLoaderConfig(YAML::Node config)
{
    ::eds::DataLoaderConfig dt_config;

    /** Read the number of events to read **/
    dt_config.num_events = config["num_events"].as<size_t>();
    dt_config.overlap = config["overlap"].as<double>(); //in percentage
    if (dt_config.overlap < 0.0) dt_config.overlap = 0.0;
    dt_config.overlap /= 100.0;

    return dt_config;
}

::eds::calib::CameraInfo Task::readCameraCalib(YAML::Node cam_calib)
{
    ::eds::calib::CameraInfo cam_info;

    cam_info.width = cam_calib["resolution"][0].as<uint16_t>();
    cam_info.height = cam_calib["resolution"][1].as<uint16_t>();
    cam_info.distortion_model = cam_calib["distortion_model"].as<std::string>();
    cam_info.D = cam_calib["distortion_coeffs"].as<std::vector<double>>();
    cam_info.intrinsics = cam_calib["intrinsics"].as<std::vector<double>>();
    if (cam_calib["flip"])
    {
        cam_info.flip = cam_calib["flip"].as<bool>();
    }
    else
        cam_info.flip  = false;

    /** T cam imu extrinsincs **/
    for (int row=0; row<4; ++row)
    {
        for (int col=0; col<4; ++col)
        {
            if (cam_calib["T_cam_imu"])
                cam_info.T_cam_imu.push_back(cam_calib["T_cam_imu"][row][col].as<double>());
        }
    }

    /** Projection matrix **/
    for (int row=0; row<3; ++row)
    {
        for (int col=0; col<4; ++col)
        {
            if (cam_calib["P"])
                cam_info.P.push_back(cam_calib["P"][row][col].as<double>());
        }
    }

    /** Rectification matrix **/
    for (int row=0; row<3; ++row)
    {
        for (int col=0; col<3; ++col)
        {
            if (cam_calib["R"])
                cam_info.R.push_back(cam_calib["R"][row][col].as<double>());
        }
    }

    /** Out resolution size **/
    if (cam_calib["resolution_out"])
    {
        cam_info.out_width = cam_calib["resolution_out"][0].as<uint16_t>();
        cam_info.out_height = cam_calib["resolution_out"][1].as<uint16_t>();
    }
    else
    {
        cam_info.out_width = 0.0;
        cam_info.out_height = 0.0;
    }

    return cam_info;
}

::base::Transform3d Task::readInitPose(YAML::Node cam_calib)
{
    ::base::Transform3d T_init; T_init.setIdentity();
    std::vector<double> translation, q;
    if (cam_calib["T_w_cam"])
    {
        translation = cam_calib["T_w_cam"]["translation"].as<std::vector<double>>();
        q = cam_calib["T_w_cam"]["orientation"].as<std::vector<double>>();

        T_init.rotate(Eigen::Quaterniond(q[3], q[0], q[1], q[2])); //Eigen constructor is w, x, y, z
        T_init.translation() = Eigen::Vector3d(translation.data());
    }

    return T_init;
}

cv::Mat Task::depthmapUndistort(const ::eds::calib::CameraInfo &cam_info, const cv::Mat &img)
{
        /** Get the cam_info data**/
        cv::Mat K, D, R_rect, P;
        K = cv::Mat_<double>::eye(3, 3);
        K.at<double>(0,0) = cam_info.intrinsics[0];
        K.at<double>(1,1) = cam_info.intrinsics[1];
        K.at<double>(0,2) = cam_info.intrinsics[2];
        K.at<double>(1,2) = cam_info.intrinsics[3];

        D = cv::Mat_<double>::zeros(4, 1);
        for (size_t i=0; i<cam_info.D.size(); ++i)
        {
            D.at<double>(i, 0) = cam_info.D[i];
        }

        if (cam_info.P.size() == 12)
        {
            P = cv::Mat_<double>::zeros(4, 4);
            for (auto row=0; row<P.rows; ++row)
            {
                for (auto col=0; col<P.cols; ++col)
                {
                    P.at<double>(row, col) = cam_info.P[(P.cols*row)+col];
                }
            }
        }

        if (cam_info.R.size() == 9)
        {
            R_rect  = cv::Mat_<double>::eye(3, 3);
            for (auto row=0; row<R_rect.rows; ++row)
            {
                for (auto col=0; col<R_rect.cols; ++col)
                {
                    R_rect.at<double>(row, col) = cam_info.R[(R_rect.cols*row)+col];
                }
            }
        }

        /** Undistort the image **/
        cv::Mat K_ref, img_undis;
        cv::Size size = img.size();
        if (P.total()>0)
            K_ref = P(cv::Rect(0,0,3,3));
        if (K.total() > 0 && D.total() > 0)
        {
            if (K_ref.total() == 0)
            {
                if (cam_info.distortion_model.compare("equidistant") != 0)
                {
                    /** radtan model **/
                    K_ref = cv::getOptimalNewCameraMatrix(K, D, cv::Size(size.width, size.height), 0.0);
                }
                else
                {
                    /** Kalibr equidistant model is opencv fisheye **/
                    cv::fisheye::estimateNewCameraMatrixForUndistortRectify(K, D, cv::Size(size.width, size.height), R_rect, K_ref);
                }
            }

            if (cam_info.distortion_model.compare("equidistant") != 0)
            {
                /** radtan model **/
                std::cout<<"[DEPTH_MAP] undistort radtan model"<<std::endl;
                cv::undistort(img, img_undis, K, D, K_ref);
            }
            else
            {
                /** Kalibr equidistant model is opencv fisheye **/
                std::cout<<"[DEPTH_MAP] equidistant radtan model"<<std::endl;
                cv::fisheye::undistortImage(img, img_undis, K, D, K_ref);
            }
        }
        else
        {
            img_undis = img;
        }

        /** Resize the image **/
        cv::Size out_size = cv::Size(frame_cam_info.out_width, frame_cam_info.out_height);
        cv::resize(img_undis, img_undis, out_size, cv::INTER_CUBIC);

    return img_undis;
}

void Task::imuMeanUnitVector(std::vector<::base::samples::IMUSensors> &data,
                            const ::base::Transform3d &T_cam_imu, const ::Eigen::Matrix3d &R_w_cam,
                            ::base::Vector3d &gyros, ::base::Vector3d &acc)
{
    std::cout<<"** [IMU_MEAN_UNIT_VECTOR] size[" <<data.size() <<"] ";
    float delta_t = ::base::NaN<double>();
    if (data.size()>0)
    {
        auto it_start = data.begin();
        auto it_end = data.begin();
        std::advance(it_end, data.size()-1);
        delta_t = (it_end->time.toSeconds() - it_start->time.toSeconds());
        std::cout<<"t_start ["<<it_start->time.toSeconds()<<"] ";
        std::cout<<"t_end ["<<it_end->time.toSeconds()<<"] ";
        std::cout<<"delta_t ["<<delta_t<<"]"<<std::endl;
    }
    else
    {
        std::cout<<"\n";
        return;
    }

    acc = ::base::Vector3d::Zero(); gyros = ::base::Vector3d::Zero();
    for (auto it : data)
    {
        gyros += it.gyro;
        acc += it.acc; 
    }

    /** Mean values in camera frame (without gravity) **/
    gyros /= data.size(); acc /= data.size();
    Eigen::Vector3d gravity(0.0, 0.0, GRAVITY);
    gyros = T_cam_imu * gyros; acc = (T_cam_imu * acc) - (R_w_cam.transpose() * gravity);
    if (!::base::isNaN(delta_t)) acc *=  delta_t;
    std::cout<<"** [IMU_MEAN_UNIT_VECTOR] ax ["<<acc[0]<<","<<acc[1]<<","<<acc[2]<<"] wx["<<gyros[0]<<","<<gyros[1]<<","<<gyros[2]<<"]"<<std::endl;

    /** Unit vector **/
    gyros /= gyros.norm(); acc /= acc.norm();

    /** Remove data in buffer **/
    data.clear();
}

bool Task::eventsToImageAlignment(const std::vector<::base::samples::Event> &events_array, ::base::Transform3d &T_kf_ef)
{
    /** Keyframe to Eventframe delta pose **/
    T_kf_ef = ::base::Transform3d::Identity();

    /** Execute the tracker and get the T_kf_ef **/
    state(OPTIMIZING);
    bool success = false;
    for (int i=this->ef->event_frame.size()-1; i>=0; --i)
    {
        success = this->tracker->optimize(i, &(this->ef->event_frame[i]), T_kf_ef, ::eds::tracking::MAD);
    }

    /** Track the points and remove the ones out of the image plane **/
    std::vector<cv::Point2d> coord = this->tracker->getCoord(true);

    /** Check if we need a new KeyFrame. Alternatively you can call
     * trackPoints for the KLT tracker**/
    this->create_kf = this->createNewKeyFrame(T_kf_ef);

    #ifdef DEBUG_PRINTS
    std::cout<<"** [EDS_TASK EVENT_TO_IMG ALIGNMENT] T_kf_ef:\n"<<T_kf_ef.matrix()<<std::endl;
    #endif
    state(RUNNING);

    return success;
}

bool Task::bootstrapping(const ::base::Time &timestamp)
{
    this->init_frames++;
    std::cout<<"** [EDS_TASK FRAME] INITIALIZING INIT_FRAME COUNT: "<<this->init_frames<<std::endl;
    if (this->init_frames >= 2 && this->ef_idx > 0)
    {
        std::cout<<"** [EDS_TASK EVENTS] INITIALIZING INSERTING KF: "<<this->kf->idx<<" IN MAP"<<std::endl;
        /** Points selection refinement **/
        this->kf->pointsRefinement(this->ef->frame[0]);

        /** Insert the KF in the global map **/
        ::eds::mapping::POINTS_SELECT_METHOD method = eds::mapping::POINTS_SELECT_METHOD::NONE;
        global_map->insert(this->kf, method);

        /** Initial Photometric Bundle Adjustement (optimize) **/
        state(OPTIMIZING);
        uint64_t kf_to_marginalize;
        this->bundles->optimize(this->global_map, kf_to_marginalize, eds::bundles::MAD, false);
        state(INITIALIZING);

        /** Initialize 2D to 2D correspondences **/
        Eigen::Matrix3d R; Eigen::Vector3d t;
        std::vector<cv::Vec3d> lines;
        if (this->kf->initialStructure(this->img_frame, R, t, lines) == false)
        {
            std::cout<<"** [EDS_TASK FRAME] INITIALIZING. RETURN: NO GOOD 2D-2D POSE"<<std::endl;
            return false;
        }
        ::base::Transform3d delta_pose(R); delta_pose.translation() = t;
        if (!::base::isNaN<double>(this->pose_w_gt.position.norm()))
        {
            /** T_kf_gt = T_w_kf^-1 * T_w_gt **/
            base::Transform3d T_kf_gt = this->global_map->getKFTransform(this->kf->idx).inverse() *
                                        this->pose_w_gt.getTransform();
            this->init_scale = T_kf_gt.translation().norm();
            std::cout<<"** [EDS_TASK FRAME] INITIALIZING. INITIAL SCALE: "<<this->init_scale<<std::endl;
        }
        delta_pose.translation() *= this->init_scale;

        /** The next T_w_kf(k+1) is the optimized kf pose T_w_kf * delta pose **/
        ::base::Transform3d T_w_kf = this->global_map->getKFTransform(this->kf->idx) * delta_pose;
        std::cout<<"** [EDS_TASK EVENTS] INITIALIZING T_w_kf:\n"<<T_w_kf.matrix()<<std::endl;

        /** Create next Keyframe **/
        this->global_map->getIDepthMap(this->kf->idx, this->depthmap, true, false);
        this->kf = std::make_shared<eds::tracking::KeyFrame>(this->kf_idx, timestamp, this->img_frame,
            this->depthmap, this->frame_cam_info, this->eds_config.mapping, this->eds_config.tracker.percent_points,
            T_w_kf, cv::Size(this->frame_cam_info.out_width, this->frame_cam_info.out_height));

        /** Increment KF INDEX **/
        this->kf_idx++;

        /** Reset initialization counter to zero **/
        this->init_frames = 0;

        if(this->global_map->size() == static_cast<size_t>(this->eds_config.bundles.window_size-2))
        {
            std::cout<<"** [EDS_TASK FRAME] INITIALIZATION FINISHED WINDOW SIZE: "<<this->global_map->size()<<std::endl;

            /** Reset the Keyframe to Eventframe transformation and tracker **/
            this->pose_kf_ef.setTransform(::base::Transform3d::Identity());

            this->global_map->cleanMap(1);

            return true;
        }
    }

    std::cout<<"** [EDS_TASK FRAME] INITIALIZATION NOT FINISHED WINDOW SIZE: "<<this->global_map->size()<<std::endl;
    return false;
}

bool Task::isInitialized()
{
    /** Check parallax and the standard deviation of the current KF inverse depth **/
    double mean, std_dev;
    this->kf->inv_depth.meanIDepth(mean, std_dev);

    ::base::Transform3d T_init_current = this->T_init.inverse() * this->kf->getPose();
    double parallax = T_init_current.translation().norm();

    std::cout<<"[EDS_TASK] INITIALIZED std_dev: "<<std_dev<< " parallax: " <<parallax<<std::endl;

    return std_dev > 0 && parallax > 0.1 && this->global_map->size() == this->eds_config.bundles.window_size;
}

bool Task::createNewKeyFrame(const ::base::Transform3d &T_kf_ef)
{
    double mean, std_dev;
    this->kf->meanResiduals(mean, std_dev);
    std::cout<<"[EDS_TASK] Event-to-Frame alignment. Mean residual: "<<mean<<std::endl;
    double m_inv_depth, third_q;
    this->kf->inv_depth.medianIDepth(m_inv_depth, third_q);
    float percent = (this->global_map->size() >= this->eds_config.bundles.window_size)? 0.1 : 0.05;
    //return false; //NOTE: uncoment this for only one KF test
    //return this->kf->needNewKFImageCriteria((this->eds_config.tracker.percent_points/100.0) * 0.8); //NOTE: uncoment this for change criteria
    return eds::utils::keyframe_selection_occlusion (T_kf_ef, 1.0/m_inv_depth, percent) || this->kf->needNewKF(percent);
}

void Task::outputEventFrameViz()
{
    /** Prepare output port image **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> event_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    event_img.reset(img);
    img = nullptr;

    /** Get the event frame visualization **/
    cv::Mat event_viz = this->ef->getEventFrameViz(false);
    //cv::Mat event_viz = this->kf->eventsOnKeyFrameViz(this->ef->frame[0]);
    //cv::Mat event_viz = this->ef->pyramidViz(false);
    /** Event frame with Epilines **/
    //cv::Mat event_viz = this->ef->epilinesViz(this->kf->coord, this->tracker->getFMatrix(), 100);

    /** Write min and max values on image **/
    double min = * std::min_element(std::begin(this->ef->event_frame[0]), std::end(this->ef->event_frame[0]));
    double max = * std::max_element(std::begin(this->ef->event_frame[0]), std::end(this->ef->event_frame[0]));
    std::string text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(event_viz, text, cv::Point(5, event_viz.rows-5), 
    cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,255,255), 0.1, cv::LINE_AA);

    /** Write in the output port **/
    ::base::samples::frame::Frame *event_img_ptr = event_img.write_access();
    event_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(event_viz, *event_img_ptr);
    event_img.reset(event_img_ptr);
    event_img_ptr->time = this->ef->time;
    _event_frame.write(event_img);

    /** Prepare output port image **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> residuals_img;
    img = new ::base::samples::frame::Frame();
    residuals_img.reset(img);
    img = nullptr;

    /** Residuals **/
    cv::Mat residuals_viz = this->kf->residualsViz();

    /** Write min and max values on image **/
    min = * std::min_element(std::begin(this->kf->residuals), std::end(this->kf->residuals));
    max = * std::max_element(std::begin(this->kf->residuals), std::end(this->kf->residuals));
    text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(residuals_viz, text, cv::Point(5, residuals_viz.rows-5), 
        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255), 0.1, cv::LINE_AA);

    /** Write in the output port **/
    ::base::samples::frame::Frame *residuals_img_ptr = residuals_img.write_access();
    frame_helper::FrameHelper::copyMatToFrame(residuals_viz, *residuals_img_ptr);
    residuals_img.reset(residuals_img_ptr);
    residuals_img_ptr->time = this->ef->time;
    _residuals_frame.write(residuals_img);
}

void Task::outputTrackerInfo(const ::base::Time &timestamp)
{
    this->tracker_info = this->tracker->getInfo();
    this->tracker_info.time = timestamp;
    _tracker_info.write(this->tracker_info);
}

void Task::outputBundlesInfo(const ::base::Time &timestamp)
{
    this->bundles_info = this->bundles->getInfo();
    this->bundles_info.time = timestamp;
    _bundles_info.write(this->bundles_info);
}

void Task::outputGradients(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp)
{
    std::cout<<"** [EDS_TASK] OUTPUT GRADIENTS_X"<<std::endl;
    /** Gradient along x-axis **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> grad_x_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    grad_x_img.reset(img);

    std::vector<cv::Point2d> coord = this->tracker->getCoord();
    cv::Mat grad_x_viz = keyframe->viz(keyframe->getGradient_x(coord, "bilinear"), false);

    ::base::samples::frame::Frame *grad_x_img_ptr = grad_x_img.write_access();
    grad_x_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(grad_x_viz, *grad_x_img_ptr);
    grad_x_img.reset(grad_x_img_ptr);
    grad_x_img_ptr->time = timestamp;
    _grad_x_frame.write(grad_x_img);

    std::cout<<"** [EDS_TASK] OUTPUT GRADIENTS_Y"<<std::endl;
    /** Gradient along y-axis **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> grad_y_img;
    img = new ::base::samples::frame::Frame();
    grad_y_img.reset(img);
 
    cv::Mat grad_y_viz = keyframe->viz(keyframe->getGradient_y(coord, "bilinear"), false);

    ::base::samples::frame::Frame *grad_y_img_ptr = grad_y_img.write_access();
    grad_y_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(grad_y_viz, *grad_y_img_ptr);
    grad_y_img.reset(grad_y_img_ptr);
    grad_y_img_ptr->time = keyframe->time;
    _grad_y_frame.write(grad_y_img);

    std::cout<<"** [EDS_TASK] OUTPUT GRADIENT_MAG"<<std::endl;
    /** Gradient magnitude **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> grad_mag;
    img = new ::base::samples::frame::Frame();
    grad_mag.reset(img);

    cv::Mat grad_mag_mat = keyframe->getGradientMagnitude(coord, "bilinear");
    cv::Mat grad_mag_viz = keyframe->viz(grad_mag_mat, true);

    /** Write min and max values on grad magnigute image **/
    double min, max;
    cv::minMaxLoc(grad_mag_mat, &min, &max);
    std::string text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(grad_mag_viz, text, cv::Point(5, grad_mag_viz.rows-5), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255), 0.1, cv::LINE_AA);

    ::base::samples::frame::Frame *grad_mag_ptr = grad_mag.write_access();
    grad_mag_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(grad_mag_viz, *grad_mag_ptr);
    grad_mag.reset(grad_mag_ptr);
    grad_mag_ptr->time = keyframe->time;
    _mag_frame.write(grad_mag);
}

void Task::outputModel(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp)
{
    std::cout<<"** [EDS_TASK] OUTPUT MODEL"<<std::endl;
    /** Gradient magnitude **/
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> model;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    model.reset(img);

    //std::vector<cv::Point2d> coord = this->tracker->getCoord();
    cv::Mat model_img = keyframe->getModel(this->tracker->linearVelocity(),
                                    this->tracker->angularVelocity(), "bilinear", 0.0);
    cv::Mat model_viz = keyframe->viz(model_img, false);
    //cv::circle(model_viz, cv::Point2d(keyframe->coord[1000]), 2.0, cv::Scalar(0, 0, 255), cv::FILLED, cv::LINE_8);

    /** Write min and max values on model image **/
    double min, max;
    cv::minMaxLoc(model_img, &min, &max);
    std::string text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(model_viz, text, cv::Point(5, model_viz.rows-5), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255), 0.1, cv::LINE_AA);

    ::base::samples::frame::Frame *model_ptr = model.write_access();
    model_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(model_viz, *model_ptr);
    model.reset(model_ptr);
    model_ptr->time = timestamp;
    _model_frame.write(model);
}

void Task::outputInvDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const base::Transform3d &T_kf_ef,
                        const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> inv_depth_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    inv_depth_img.reset(img);
 
    /** Real depth **/
    //keyframe->inv_depth.update(T_kf_ef, keyframe->coord, keyframe->tracks);
    /** Debug Test code **/
    //std::vector<cv::Point2d> coord = tracker->getCoord(false);//coord in event frame
    //keyframe->inv_depth.update(T_kf_ef, keyframe->coord, coord);

    /** Get the inverse depth map viz **/
    std::vector<double> idp = keyframe->inv_depth.getIDepth();
    cv::Mat inv_depth_viz = keyframe->idepthmapViz(keyframe->coord, idp, "nn", 0.0);

    /** Write min and max values on image **/
    double min = * std::min_element(std::begin(idp), std::end(idp));
    double max = * std::max_element(std::begin(idp), std::end(idp));
    std::string text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(inv_depth_viz, text, cv::Point(5, inv_depth_viz.rows-5), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(0,0,0), 0.1, cv::LINE_AA);

    /** Output inverse depth map **/
    ::base::samples::frame::Frame *inv_depth_img_ptr = inv_depth_img.write_access();
    inv_depth_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(inv_depth_viz, *inv_depth_img_ptr);
    inv_depth_img.reset(inv_depth_img_ptr);
    inv_depth_img_ptr->time = timestamp;
    _inv_depth_frame.write(inv_depth_img);

    /** Output the Local map **/
    std::vector<double>model = keyframe->getSparseModel(this->tracker->linearVelocity(), this->tracker->angularVelocity());
    ::base::samples::Pointcloud point_cloud = keyframe->getMap(idp, model, ::eds::tracking::MAP_COLOR_MODE::EVENTS);
    _local_map.write(point_cloud);
}

void Task::outputSigmaInvDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> sigma_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    sigma_img.reset(img);
 
    /** Get the sigma map (uncertainty) viz for the inverse depth **/
    double min, max;
    //cv::Mat sigma_viz = keyframe->inv_depth.sigmaViz(keyframe->img, keyframe->coord, min, max);
    cv::Mat sigma_viz = keyframe->weightsViz(min, max);
    std::string text = "min: " + std::to_string(min) + " max: " + std::to_string(max);
    cv::putText(sigma_viz, text, cv::Point(5, sigma_viz.rows-5), 
                cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255), 0.1, cv::LINE_AA);


    /** Output inverse depth map **/
    ::base::samples::frame::Frame *sigma_img_ptr = sigma_img.write_access();
    sigma_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(sigma_viz, *sigma_img_ptr);
    sigma_img.reset(sigma_img_ptr);
    sigma_img_ptr->time = timestamp;
    _sigma_depth_frame.write(sigma_img);
}

void Task::outputConvergenceDepth(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> converg_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    converg_img.reset(img);
 
    /** Get convergence for the inverse depth **/
    cv::Mat converg_viz = keyframe->inv_depth.convergenceViz(keyframe->img, keyframe->coord);

    /** Output inverse depth map **/
    ::base::samples::frame::Frame *converg_img_ptr = converg_img.write_access();
    converg_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(converg_viz, *converg_img_ptr);
    converg_img.reset(converg_img_ptr);
    converg_img_ptr->time = timestamp;
    _converge_depth_frame.write(converg_img);
}

void Task::outputGlobalMapMosaic(const std::shared_ptr<::eds::mapping::GlobalMap> &global_map, const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> mosaic_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    mosaic_img.reset(img);

    /** Get the mosaic of keyframes **/
    cv::Mat mosaic = global_map->vizMosaic(this->eds_config.bundles.window_size);

    /** Output the mosaic of global map keyframes **/
    ::base::samples::frame::Frame *mosaic_img_ptr = mosaic_img.write_access();
    mosaic_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(mosaic, *mosaic_img_ptr);
    mosaic_img.reset(mosaic_img_ptr);
    mosaic_img_ptr->time = timestamp;
    _keyframes_frame.write(mosaic_img);
}

void Task::outputOpticalFlowFrameViz(const std::shared_ptr<::eds::tracking::KeyFrame> &keyframe, const ::base::Time &timestamp)
{
    RTT::extras::ReadOnlyPointer<base::samples::frame::Frame> of_img;
    ::base::samples::frame::Frame *img = new ::base::samples::frame::Frame();
    of_img.reset(img);

    /** DEBUG: test for the GT optical flow **/
    //std::vector<Eigen::Vector2d> tracks;
    //std::vector<cv::Point2d> coord = tracker->getCoord(false);
    //auto it_w = coord.begin();
    //auto it_c = keyframe->coord.begin();
    //for (; it_w != coord.end() && it_c != keyframe->coord.end(); ++it_w, ++it_c)
    //{
    //    tracks.push_back(Eigen::Vector2d((*it_w).x-(*it_c).x, (*it_w).y-(*it_c).y));
    //}
    /** Get the optical flow **/
    cv::Mat optical_flow = eds::utils::flowArrowsOnImage(keyframe->img, keyframe->coord,
                                                keyframe->tracks, cv::Vec3b(0.0, 255.0, 0.0) /*BGR*/, 10);

    /** Output the optical flow image **/
    ::base::samples::frame::Frame *of_img_ptr = of_img.write_access();
    of_img_ptr->image.clear();
    frame_helper::FrameHelper::copyMatToFrame(optical_flow, *of_img_ptr);
    of_img.reset(of_img_ptr);
    of_img_ptr->time = timestamp;
    _of_frame.write(of_img);
}

void Task::outputPoseKFs()
{
    /** Write sliding window KF poses port: T_w_kf[i] **/
    eds::VectorKFs pose_kfs;
    this->global_map->getKFPoses(pose_kfs.kfs);
    if (!pose_kfs.kfs.empty())
    {
        pose_kfs.time = pose_kfs.kfs[pose_kfs.kfs.size()-1].time;
        _pose_w_kfs.write(pose_kfs);
    }
}

void Task::outputKeyFrameViz()
{
    /** Output mosaic **/
    this->outputGlobalMapMosaic(this->global_map, this->kf->time);
}

void Task::outputGlobalMap()
{
    ::base::samples::Pointcloud pcl_map;
    this->global_map->getMap(pcl_map, true, true);
    _global_map.write(pcl_map);
}
