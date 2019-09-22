define SERIAL_DELAY 6  // micro sec
#            if __GNUC__ < 6
#                define READ_WRITE_START_ADJUST 30  // cycles
#                define READ_WRITE_WIDTH_ADJUST 3   // cycles
#            else
#                define READ_WRITE_START_ADJUST 33  // cycles
#                define READ_WRITE_WIDTH_ADJUST 7   // cycles
#            endif