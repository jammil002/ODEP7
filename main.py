import numpy as np
import matplotlib.pyplot as plt
import sys
print(sys.executable)


def utilizationRatio(lam, mu):
    return lam / mu


def systemThroughput(lam, mu):
    return min(lam, mu)


def meanNumberInSystem(lam, mu, expectedTime):
    return lam * expectedTime


def meanTimeInSystem(expectedNumber, lam):
    return expectedNumber / lam


def calculateLorenzAttractor(x, y, z, rho, sigma=10, beta=2.667):
    xDot = sigma * (y - x)
    yDot = rho * x - y - x * z
    zDot = x * y - beta * z
    return xDot, yDot, zDot


def simulateAndVisualizeLorenzAttractor(rho):
    timeStep = 0.01  # Time step
    numSteps = 10000  # Number of steps
    xCoordinates = np.empty(numSteps + 1)
    yCoordinates = np.empty(numSteps + 1)
    zCoordinates = np.empty(numSteps + 1)
    xCoordinates[0], yCoordinates[0], zCoordinates[0] = (7.5, 22.5, 35)  # Initial conditions

    for i in range(numSteps):
        xDot, yDot, zDot = calculateLorenzAttractor(xCoordinates[i], yCoordinates[i], zCoordinates[i], rho)
        xCoordinates[i + 1] = xCoordinates[i] + (xDot * timeStep)
        yCoordinates[i + 1] = yCoordinates[i] + (yDot * timeStep)
        zCoordinates[i + 1] = zCoordinates[i] + (zDot * timeStep)

    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')
    axis.plot(xCoordinates, yCoordinates, zCoordinates, lw=0.5, color='purple')
    axis.set_xlabel("X Axis")
    axis.set_ylabel("Y Axis")
    axis.set_zlabel("Z Axis")
    axis.set_title("Lorenz Attractor for ρ = " + str(rho))
    plt.show()


# Main execution block
if __name__ == '__main__':
    simulateAndVisualizeLorenzAttractor(24)

    # Initialize data and perform simulation
    dataPoints = [(1, 2.22), (2, 1.76), (3, 2.13), (4, 0.14), (5, 0.76), (6, 0.70), (7, 0.47),
                  (8, 0.22), (9, 0.18), (10, 2.41), (11, 0.41), (12, 0.46), (13, 1.37),
                  (14, 0.27), (15, 0.27)]
    arrivalTimes = [entry[0] for entry in dataPoints]
    serviceDurations = [entry[1] for entry in dataPoints]
    departureTimes = [arrivalTimes[0] + serviceDurations[0]]
    serviceStartTimes = [arrivalTimes[0]]
    waitingTimes = [max(0, departureTimes[0] - arrivalTimes[0] - serviceDurations[0])]
    numCustomersInSystem = [1]
    numCustomersInQueue = [0]

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

    plt.figure(figsize=(15, 12))
    titles = ['Arrival Time vs Service Start Time', 'Arrival Time vs Exit Time',
              'Arrival Time vs Time in Queue', 'Arrival Time vs Number in System',
              'Arrival Time vs Number in Queue']
    dataFunctions = [serviceStartTimes, departureTimes, waitingTimes, numCustomersInSystem, numCustomersInQueue]
    yLabels = ['Service Start Time', 'Exit Time', 'Time in Queue', 'Number of Customers in System',
               'Number of Customers in Queue']
    for i in range(5):
        plt.subplot(3, 2, i + 1)
        plt.plot(arrivalTimes, dataFunctions[i], marker='o', lw=0.5, color='purple')
        plt.xlabel('Arrival Time')
        plt.ylabel(yLabels[i])
        plt.title(titles[i])
    plt.tight_layout()
    plt.show()

    # Metrics for system
    totalTime = departureTimes[-1]
    averageQueueLength = sum(waitingTimes) / totalTime
    averageQueueLengthPerArrival = sum(waitingTimes) / arrivalTimes[-1]

    # Print computed metrics
    print("L_q:", averageQueueLength)
    print("L_q(A):", averageQueueLengthPerArrival)
    print("\n")

    # Visualization of the impact of rate changes
    rateIncreaseFactor = 2
    arrivalRate = 0.5
    serviceRate = 1.0
    updatedArrivalRate = rateIncreaseFactor * arrivalRate
    updatedServiceRate = rateIncreaseFactor * serviceRate
    originalUtilization = utilizationRatio(arrivalRate, serviceRate)
    originalThroughput = systemThroughput(arrivalRate, serviceRate)
    originalMeanNumberInSystem = 1 / (serviceRate - arrivalRate)
    originalMeanTimeInSystem = originalMeanNumberInSystem / arrivalRate
    updatedUtilization = utilizationRatio(updatedArrivalRate, updatedServiceRate)
    updatedThroughput = systemThroughput(updatedArrivalRate, updatedServiceRate)
    updatedMeanNumberInSystem = meanNumberInSystem(updatedArrivalRate, updatedServiceRate, originalMeanTimeInSystem)
    updatedMeanTimeInSystem = meanTimeInSystem(updatedMeanNumberInSystem, updatedArrivalRate)

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
    plt.show()

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