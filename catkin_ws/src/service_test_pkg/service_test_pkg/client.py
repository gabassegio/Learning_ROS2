import sys
from example_interfaces.srv import AddTwoInts
import rclpy
from rclpy.node import Node

class client_class(Node):
    def __init__(self):
        super().__init__('client_class')
        self.cli = self.create_client(AddTwoInts,'add_two_ints')
        while not self.cli.wait_for_service(timeout_sec=1):
            self.get_logger().info('Service not available.. Try again')
        self.req = AddTwoInts.Request()
    def send_request(self,a,b):
        self.req.a = a
        self.req.b = b 
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self,self.future)
        return self.future.result()

def main(args=None):
    rclpy.init(args=args)
    client = client_class()
    response = client.send_request(int(sys.argv[1]), int(sys.argv[2]))
    client.get_logger().info(
        'Result of add_two_ints: for %d + %d = %d' %
        (int(sys.argv[1]), int(sys.argv[2]), response.sum))
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

