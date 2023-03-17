#include <iostream>
#include <boost/program_options.hpp>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <set>
#include <cmath>
#include <vector>
#include <math.h>

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/compression/octree_pointcloud_compression.h>
#include <pcl/visualization/cloud_viewer.h>

// exsampe 
// ./pcl_pcc --input_ply_path= --bin_path= --ply_path= --model=
using namespace pcl;
using namespace std;
namespace po = boost::program_options;

int main(int argc,char** argv) {
	// Declare the input
	string input_ply;
	string bin_path;
	string ply_path;
        // Declare the supported options.
        po::options_description desc("Program options");
        desc.add_options()
                //Options
                ("input_ply_path", po::value<string>(&input_ply)->required(), "the input of ply data path")
                ("bin_path", po::value<string>(&bin_path)->required(), "the path of bin data save path")
                ("ply_path", po::value<string>(&ply_path)->required(), "the path of re-ply data save path")
                ;
        // Parse the command line
        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        // Print help
        if (vm.count("help")){
                cout << desc << "\n";
                return false;
        }
        // Process options.
        po::notify(vm);

	// data load
    pcl::PointCloud<pcl::PointXYZ> in_PC;
    pcl::PLYReader reader;
    if (pcl::io::loadPLYFile(input_ply, in_PC) == -1) {
        PCL_ERROR("Failed to load PLYFile!");
        return -1;
    }
    pcl::io::savePLYFileASCII (input_ply, in_PC);
    // encode
    /*
     *  model:
     *  LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
     *  MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
     *  HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR
     *  LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
     *  MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
     *  HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR
     */
    bool showStatistics = true;
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    /*
    if (pcc_model == 1) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::LOW_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    } else if (pcc_model == 2) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::MED_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    } else if (pcc_model == 3) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::HIGH_RES_ONLINE_COMPRESSION_WITHOUT_COLOR;
    } else if (pcc_model == 4) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::LOW_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR;
    } else if (pcc_model == 5) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::MED_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR;
    } else if (pcc_model == 6) {
    pcl::io::compression_Profiles_e compressionProfile =
            pcl::io::HIGH_RES_OFFLINE_COMPRESSION_WITHOUT_COLOR;
    }
    */
    pcl::io::OctreePointCloudCompression<pcl::PointXYZ>* PointCloudEncoder;
    PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compressionProfile, showStatistics);
    std::stringstream compressedData;
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_PC(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudEncoder->encodePointCloud(in_PC.makeShared(), compressedData);
    ofstream OutbitstreamFile(bin_path, fstream::binary | fstream::out);
    OutbitstreamFile << compressedData.str();
    OutbitstreamFile.close();
    ifstream bitstreamFile(bin_path, ios::binary);
    //decode
    PointCloudEncoder->decodePointCloud(bitstreamFile, out_PC);
    bitstreamFile.close();
    pcl::io::savePLYFileASCII(ply_path, *out_PC);

	return 0;


}

