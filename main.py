import numpy as np
import matplotlib.pyplot as plt


def lorenz(x, y, z, r, s=10, b=2.667):
    x_dot = s * (y - x)
    y_dot = r * x - y - x * z
    z_dot = x * y - b * z
    return x_dot, y_dot, z_dot


def simulateAndVisualizeLorenzAttractor(r):
    dt = 0.01  # Time step
    num_steps = 10000  # Number of steps
    # Initialize arrays for x, y, z
    xs = np.empty(num_steps + 1)
    ys = np.empty(num_steps + 1)
    zs = np.empty(num_steps + 1)
    xs[0], ys[0], zs[0] = (7.5, 22.5, 35)  # Initial conditions

    # Simulate the Lorenz system
    for i in range(num_steps):
        x_dot, y_dot, z_dot = lorenz(xs[i], ys[i], zs[i], r)
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    # Create a figure and a 3D subplot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs, ys, zs, lw=0.5, color='purple')
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor: r = " + str(r))
    plt.show()


simulateAndVisualizeLorenzAttractor(28)

# Initialize data
dataPoints = [
    (1, 2.22), (2, 1.76), (3, 2.13), (4, 0.14), (5, 0.76), (6, 0.70), (7, 0.47),
    (8, 0.22), (9, 0.18), (10, 2.41), (11, 0.41), (12, 0.46), (13, 1.37),
    (14, 0.27), (15, 0.27)
]

# Extract data
arrivalTimes = [entry[0] for entry in dataPoints]
serviceDurations = [entry[1] for entry in dataPoints]

# Initialize lists for calculations
departureTimes = [arrivalTimes[0] + serviceDurations[0]]
serviceStartTimes = [arrivalTimes[0]]
waitingTimes = [max(0, departureTimes[0] - arrivalTimes[0] - serviceDurations[0])]
numCustomersInSystem = [1]
numCustomersInQueue = [0]

# Compute times and customer counts
for i in range(1, len(arrivalTimes)):
    currentServiceStartTime = max(departureTimes[i - 1], arrivalTimes[i])
    serviceStartTimes.append(currentServiceStartTime)
    departureTimes.append(currentServiceStartTime + serviceDurations[i])
    currentWaitingTime = max(0, departureTimes[i] - arrivalTimes[i] - serviceDurations[i])
    waitingTimes.append(currentWaitingTime)
    currentNumCustomers = np.sum(np.array(arrivalTimes) <= arrivalTimes[i]) - np.sum(
        np.array(departureTimes) <= arrivalTimes[i])
    numCustomersInSystem.append(currentNumCustomers)
    numDepartures = np.sum(np.array(departureTimes) <= arrivalTimes[i])
    numCustomersInQueue.append(i - numDepartures)

# Metrics for system
totalTime = departureTimes[-1]
averageQueueLength = sum(waitingTimes) / totalTime
averageQueueLengthPerArrival = sum(waitingTimes) / arrivalTimes[-1]

# Plotting section
plt.figure(figsize=(15, 12))

# Subplots for various relationships
titles = ['Customer Arrival Time vs Service Start Time', 'Customer Arrival Time vs Exit Time',
          'Customer Arrival Time vs Time in Queue', 'Customer Arrival Time vs Number of Customers in System',
          'Customer Arrival Time vs Number of Customers in Queue']
dataFunctions = [serviceStartTimes, departureTimes, waitingTimes, numCustomersInSystem, numCustomersInQueue]
yLabels = ['Service Start Time', 'Exit Time', 'Time in Queue', 'Number of Customers in System',
           'Number of Customers in Queue']

for i in range(5):
    plt.subplot(3, 2, i + 1)
    plt.plot(arrivalTimes, dataFunctions[i], marker='o')
    plt.xlabel('Arrival Time')
    plt.ylabel(yLabels[i])
    plt.title(titles[i])

plt.tight_layout()
plt.show()

# Print computed metrics
print("L_q:", averageQueueLength)
print("L_q(A):", averageQueueLengthPerArrival)
print("\n")

# Print headers with proper spacing
header = "| {0:<12} | {1:<16} | {2:<18} | {3:<15} | {4:<13} | {5:<22} | {6:<22} |".format(
    "Arrival Time", "Service Duration", "Service Start Time", "Departure Time", "Waiting Time",
    "Customers in System", "Customers in Queue"
)
print(header)
print("-" * len(header))  # Print a dividing line of dashes under the header

# Print data rows
for i in range(len(dataPoints)):
    print(
        "| {0:<12} | {1:<16.2f} | {2:<18.2f} | {3:<15.2f} | {4:<13.2f} | {5:<22.2f} | {6:<22.2f} |".format(
            dataPoints[i][0], dataPoints[i][1], serviceStartTimes[i], departureTimes[i],
            waitingTimes[i], numCustomersInSystem[i], numCustomersInQueue[i]
        )
    )


# Function definitions for operations research calculations
def utilizationRatio(lam, mu):
    return lam / mu


def systemThroughput(lam, mu):
    return min(lam, mu)


def meanNumberInSystem(lam, mu, expectedTime):
    return lam * expectedTime


def meanTimeInSystem(expectedNumber, lam):
    return expectedNumber / lam


# Parameters for rate calculations
arrivalRate = 0.5
serviceRate = 1.0
rateIncreaseFactor = 2

# Calculate new rates
updatedArrivalRate = rateIncreaseFactor * arrivalRate
updatedServiceRate = rateIncreaseFactor * serviceRate

# Original system values
originalUtilization = utilizationRatio(arrivalRate, serviceRate)
originalThroughput = systemThroughput(arrivalRate, serviceRate)
originalMeanNumberInSystem = 1 / (serviceRate - arrivalRate)
originalMeanTimeInSystem = originalMeanNumberInSystem / arrivalRate

# Updated system values
updatedUtilization = utilizationRatio(updatedArrivalRate, updatedServiceRate)
updatedThroughput = systemThroughput(updatedArrivalRate, updatedServiceRate)
updatedMeanNumberInSystem = meanNumberInSystem(updatedArrivalRate, updatedServiceRate, originalMeanTimeInSystem)
updatedMeanTimeInSystem = meanTimeInSystem(updatedMeanNumberInSystem, updatedArrivalRate)

# Visualization of the impact of rate changes
labels = ['Utilization (ρ)', 'Throughput (X)', 'Mean number in system (E[N])', 'Mean time in system (E[T])']
originalValues = [originalUtilization, originalThroughput, originalMeanNumberInSystem, originalMeanTimeInSystem]
updatedValues = [updatedUtilization, updatedThroughput, updatedMeanNumberInSystem, updatedMeanTimeInSystem]

xCoords = np.arange(len(labels))
barWidth = 0.35

fig, ax = plt.subplots()
rects1 = ax.bar(xCoords - barWidth / 2, originalValues, barWidth, color='purple', label='Original')
rects2 = ax.bar(xCoords + barWidth / 2, updatedValues, barWidth, color='black', label='Updated')

ax.set_ylabel('Values')
ax.set_title('Effect of Increasing Arrival and Service Rates')
ax.set_xticks(xCoords)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
