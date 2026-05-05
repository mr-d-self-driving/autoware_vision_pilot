#include <iostream>
#include <models/auto_drive.hpp>
#include <can_interface/can_reader.hpp>
#include <can_interface/can_writer.hpp>
#include <visualization/visualization.hpp>

int main() {
    std::cout << "Hello and welcome to  VisionPilot!\n";

    can_reader::CanReader reader;
    std::cout << reader.read() << std::endl;

    return 0;
}
