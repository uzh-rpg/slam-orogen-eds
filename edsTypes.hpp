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

#ifndef EDS_TYPES_HPP
#define EDS_TYPES_HPP


#include <eds/tracking/Config.hpp>
#include <eds/mapping/Config.hpp>
#include <eds/bundles/Config.hpp>
#include <eds/utils/Config.hpp>

#include <base/Time.hpp>
#include <base/samples/RigidBodyState.hpp>

/* If you need to define types specific to your oroGen components, define them
 * here. Required headers must be included explicitly
 *
 * However, it is common that you will only import types from your library, in
 * which case you do not need this file
 */

namespace eds {

    struct DataLoaderConfig
    {
        size_t num_events;
        double overlap;
    };

    struct EDSConfiguration
    {
        DataLoaderConfig data_loader;
        ::eds::tracking::Config tracker;
        ::eds::mapping::Config mapping;
        ::eds::bundles::Config bundles;
        ::eds::recorder::Config recorder;
    };

    struct VectorKFs
    {
        ::base::Time time;
        std::vector<base::samples::RigidBodyState> kfs;
    };

    struct EventFrameVector
    {
        ::base::Time time;
        std::vector<double> data;
    };

    struct ModelFrameVector
    {
        ::base::Time time;
        std::vector<double> data;
    };

    typedef eds::tracking::TrackerInfo TrackerInfo;
    typedef eds::bundles::PBAInfo PBAInfo;
}

#endif

