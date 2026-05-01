#ifndef INC_VISIONPILOT_CAN_READER_HPP
#define INC_VISIONPILOT_CAN_READER_HPP
#include <string>

namespace can_reader {
    class CanReader {
    public:
        CanReader() = default;

        ~CanReader() = default;

        // Simple read method
        std::string read();
    };
}

#endif //INC_VISIONPILOT_CAN_READER_HPP
