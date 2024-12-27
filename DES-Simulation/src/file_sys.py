import os
import re

SIMU_RESULT_PATH = '../data/'
SIMU_STATISTICS_PATH = '../statistics/'
SIMU_VISUALIZATION_PATH = '../visualization/'

CONTENT_TEMPLATE = ("Customer {customer_number} in system with {system_server_number} servers, "
                    "arrived at {arrive_time:.4f}, and got service after waiting {waiting_time:.4f}, service time is {service_time:.4f}\n")

CONTENT_PATTERN = r"Customer (\d+) in system with \d+ servers, arrived at (\d+\.\d+), and got service after waiting (\d+\.\d+), service time is (\d+\.\d+)"


def write(customer_number, arrive_time, waiting_time, service_time, system_server_number, filepath, lam, miu, prefix="FIFO"):
    filename = f"{prefix}_server_system_{system_server_number}_with_basic_lamb_{lam}_u_{miu}.log"
    full_path = os.path.join(filepath, filename)
    
    content = CONTENT_TEMPLATE.format(customer_number=customer_number, 
                                      system_server_number=system_server_number, 
                                      arrive_time=arrive_time, 
                                      waiting_time=waiting_time,
                                      service_time=service_time)
    
    # make sure the file exists
    if not os.path.exists(filepath):
        os.makedirs(filepath)

    with open(full_path, 'a') as file:
        file.write(content)

def read(system_server_number, filepath, lam, miu, prefix="FIFO"):
    filename = f"{prefix}_server_system_{system_server_number}_with_basic_lamb_{lam}_u_{miu}.log"
    full_path = os.path.join(filepath, filename)
    
    customers = []
    arrival_times = []
    waiting_times = []
    service_times = []
    
    pattern = re.compile(CONTENT_PATTERN)
    
    with open(full_path, 'r') as file:
        for line in file:
            match = pattern.search(line)
            if match:
                customers.append(int(match.group(1)))
                arrival_times.append(float(match.group(2)))
                waiting_times.append(float(match.group(3)))
                service_times.append(float(match.group(4)))

            else:
                print(f"Invalid line  {line}")

    return customers, arrival_times, waiting_times, service_times

def check_directory(directory):
    return os.path.exists(directory)

def create_directory(directory):
    os.makedirs(directory)

# Example Usage:
# write(1, 0.56, 1.23, 5, 'data', 0.5, 0.7)
# arrival_times, waiting_times = read(5, 'data', 0.5, 0.7)
# print(arrival_times, waiting_times)
