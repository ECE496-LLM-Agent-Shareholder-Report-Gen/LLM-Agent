import time

start_time = time.time_ns()
time.sleep(5)
stop_time = time.time_ns()

duration = stop_time - start_time
duration_s = duration/1e9

print("duration: ", duration)
print("duration_s: ", duration_s)