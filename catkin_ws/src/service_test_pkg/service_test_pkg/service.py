from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class service_class(Node):
    def __init__(self):
        super().__init__('service_class')
        self.srv = self.create_service(AddTwoInts,'add_two_ints',self.callback)
    
    def callback(self, request, response):
        response.sum = request.a + request.b
        self.get_logger().info('Incoming request\na: %d b: %d' % (request.a, request.b))
        return response

def main(args=None):
    rclpy.init(args=args)
    service = service_class()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == '__main__':
    main()