# put your routing program here!
#!/usr/bin/env python
# route.py : Route finding problem
# Rohan Kasture, 2017 created
#

#Note: The routes which have a speed limit of 0 are not considered. A speed limit of zero may imply it is under maintenance and
#you cannot use that road segment.

#(1) Which search algorithm seems to work best for each routing options?
#For long distances, A star works better than the rest.Uniform also works better but it explores more nodes
#than what A star does. DFS performs worst. BFS gives sub optimal paths.

#(2) Which algorithm is fastest in terms of the amount of computation time required by your
#program, and by how much, according to your experiments?
#If there are multiple solutions and they are dense and we dont require optimal path, DFS is best amongst all.DFS requires least
# time for finding goal node.For shorter routes it is comparable to A star. But for long routes DFS takes half or even less
# amount of time than A star. Astar takes time due to the use of heuristic to reach the goal
#For finding optimal paths A star and uniform are better


#(3) Which algorithm requires the least memory, and by how much, according to your experiments?
#DFS requires least memory than others.It requires linear space i.e O(bm) where b is branching factor and m is maximum depth
#BFS required O(b^d) as it keeps every node in fringe. Uniform and A star algorithm also requires O(b^d) space

#(4) Which heuristic function(s) did you use, how good is it,and how might you make it/them better?
# f(n) = g(n) + h(s), where g(n) is the cost to reach current node and h(s) is cost to reach goal node
# h(s) is the distance computed between current node and goal node using haversine formula
# In many cases we dont have GPS data for nodes, in that case value of h(s) will be the heuristic value of parent of s minus
# the distance travelled from this node to parent. As we moved back i.e away from goal, I have subtracted the distance.
#This heuristic will never overestimate the path to goal node. It is admissible.It will explore nodes which have
# minimum f(n) value
#This is implemented using heapq structure in python . It is min priority queue.

#Longtour function is implemented in Astar and uniform, but it will give longest tour in A Star as it uses heuristics to reach the
#goal and for longtour we will use same heuristic to move away from goal.



import  time
import sys
import math
import copy
import  heapq

def read_city_gps():            # Reading city gps into a dictionary
    file = open('city-gps.txt', 'r')
    result = {}
    for line in file.readlines():
        city_data = line.split(" ")
        result[city_data[0]] = [float(city_data[1]),float(city_data[2])]
    file.close()
    return result


def read_road_segments():            # Reading road segments
    file = open('road-segments.txt', 'r')
    result = {}
    sum = 0
    count = 0
    for line in file.readlines():
        city_data = line.split(" ")
        # Ignoring the entires which dont have distance or speed data
        if city_data[2] == "0" or city_data[2] == "" or city_data[3] == "" or city_data[3] == "0":
            continue
        sum += int(city_data[3])
        count = count + 1
        city_data[4] = city_data[4][:-1]
        if city_data[0] in result:
            result[city_data[0]]['connected'] += [city_data[1]]
            if routing_algorithm == "uniform" and result[city_data[0]]['connected'][0] == city_data[1]:
                value1 = float(result[city_data[0]][city_data[1]][0]) / float(result[city_data[0]][city_data[1]][1])
                value2 = float(city_data[2]) / float(city_data[3])
                if(cost_function == "distance" and int(result[city_data[0]][city_data[1]][0]) < int(city_data[2])):
                    pass
                elif(cost_function == "time" and value1 < value2):
                    pass
                else:
                    result[city_data[0]][city_data[1]] = [city_data[2], city_data[3], city_data[4]]  # default
            else:
                result[city_data[0]][city_data[1]] = [city_data[2], city_data[3], city_data[4]]  # default
        else:
            result[city_data[0]] = {'connected': [city_data[1]]}
            result[city_data[0]][city_data[1]] = [city_data[2],city_data[3],city_data[4]]
        if city_data[1] in result:
            result[city_data[1]]['connected'] += [city_data[0]]
            if routing_algorithm == "uniform" and result[city_data[1]]['connected'][0] == city_data[0]:
                value1 = float(result[city_data[1]][city_data[0]][0]) / float(result[city_data[1]][city_data[0]][1])
                value2 = float(city_data[2]) / float(city_data[3])
                if ( cost_function == "distance" and  int(result[city_data[1]][city_data[0]][0]) < int(city_data[2])):
                    pass
                elif (cost_function == "time" and  value1 <  value2 ):
                    pass
                else:
                    result[city_data[1]][city_data[0]] = [city_data[2], city_data[3], city_data[4]]  # default
            else:
                result[city_data[1]][city_data[0]] = [city_data[2],city_data[3],city_data[4]]
        else:
            result[city_data[1]]= {'connected': [city_data[0]]}
            result[city_data[1]][city_data[0]] = [city_data[2],city_data[3],city_data[4]]
    file.close()
    avg_speed = sum / count
    return avg_speed,result

def successors_cities_bfs_dfs(city,routing_algo, parameter,road_segments):
    result = []

    for neighbour_city in road_segments[city[0]]['connected']:
        result.append([neighbour_city, city[0],road_segments[city[0]][neighbour_city][0],road_segments[city[0]][neighbour_city][1]])
    return result

#Print the final path using backtracte function
def backtrace(parent, start, end,road_segment,cost):
    path = [end]
    x = end
    total_distance = 0
    total_time = 0
    while x != start:
        total_distance = total_distance + float(road_segment[x][parent[x]][0])
        total_time = total_time + (float(road_segment[x][parent[x]][0]) / float(road_segment[x][parent[x]][1]))
        path.append(parent[x])
        x = parent[x]
    path.reverse()
    if cost_function == "segments":
        print  "No of segments:" + str(cost)
    return [total_distance,total_time,path]

def solve_dfs_bfs(start_city, end_city,routing_algorithm, cost_function,road_segment):  # dfs/bfs
    fringe = [[start_city,'root',0]]
    parent = {}         #used to backtrack from goal node to start node
    visited_cities = []


    while len(fringe) > 0:
        if routing_algorithm == "bfs":
            current_city = fringe.pop(0)
        else:  # dfs
            current_city = fringe.pop()

        if current_city[0] == end_city:
            parent[current_city[0]] = current_city[1]
            return backtrace(parent, start_city, end_city, road_segment, float(current_city[2]))
        if current_city[0] not in visited_cities:
            visited_cities.append(current_city[0])
            parent[current_city[0]] = current_city[1]
            for s in successors_cities_bfs_dfs(current_city, routing_algorithm, cost_function, road_segment):
                if s[0] in visited_cities:
                    continue
                fringe.append(s)
                if s[0] == end_city: #Goal city check
                     parent[s[0]] = current_city[0]
                     return backtrace(parent, start_city, end_city, road_segment, float(current_city[2]))
    return False


def successors_cities_uniform(city, routing_algo, parameter,road_segments): #succ function for uniform cost
    result = []

    for neighbour_city in road_segments[city[1]]['connected']:
        if parameter == "segments":
            result.append([int(city[0]) + 1,neighbour_city, city[1]])
        elif parameter == "time":
            result.append([(float(city[0]) + float(road_segments[city[1]][neighbour_city][0]) / float(road_segments[city[1]][neighbour_city][1])),neighbour_city, city[1]])
        elif parameter == "distance": #distance
            result.append([int(city[0])+ int(road_segments[city[1]][neighbour_city][0]),neighbour_city, city[1]])
        else: # longtour cost function[
            #implemented by multiplying the distance by -1, turning min priority queue to max prirority queue
            result.append([(int(city[0]) + int(road_segments[city[1]][neighbour_city][0]))* (-1), neighbour_city, city[1]])
    return result


def solve_uniform(start_city, end_city,routing_algorithm, cost_function,road_segment):
    parent = {}
    visited_cities = []
    fringe = []
    list = [0,start_city,'root']
    heapq.heappush(fringe,list)
    current_city = []
    while len(fringe) > 0:
        current_city = heapq.heappop(fringe)
        if current_city[1] == end_city: #Goal city found
            parent[current_city[1]] = current_city[2]
            return backtrace(parent, start_city, end_city, road_segment, float(current_city[0]))
        if current_city[1] not  in visited_cities:
            visited_cities.append(current_city[1])
            parent[current_city[1]] = current_city[2]
            for s in successors_cities_uniform(current_city,routing_algorithm,cost_function,road_segment):
                if s[1] in visited_cities: #If already visited dont visit again
                    continue
                heapq.heappush(fringe,s)
    return False


def successors_cities_astar(city,routing_algo, parameter,road_segments,parent,heuristic): #succ function for astar
    #city is parent city whose successors are to be found
    result = []
    distance_successor = 0
    distance_goal = 0
    for neighbour_city in road_segments[city[1]]['connected']:
        if parameter == "segments":
            result.append([int(city[0]) + 1,neighbour_city, city[1],int(city[0]) + 1])

        elif parameter == "time":
            distance_goal = float(city[0]) + float(road_segments[city[1]][neighbour_city][0]) / float(road_segments[city[1]][neighbour_city][1])
            result.append([distance_goal,neighbour_city, city[1],distance_goal])
        elif parameter == "distance": #distance
            neighbour_city_gps = find_lat_lon(neighbour_city)
            if neighbour_city_gps == [0,0]:
                heuristic[neighbour_city] = float(heuristic[city[1]]) - float(road_segments[city[1]][neighbour_city][0])
                distance_goal = float(heuristic[neighbour_city]) + float(city[3])
            else:
                heuristic[neighbour_city] = distance_lat_lon(neighbour_city_gps[0],neighbour_city_gps[1],goal_lat_lon[0],goal_lat_lon[1])**(1./3.)
                distance_goal =  float(heuristic[neighbour_city]) + float(city[3])

            result.append([float(distance_goal),neighbour_city,city[1],float(city[3])+ float(road_segments[city[1]][neighbour_city][0])])
        else:#longtour
            neighbour_city_gps = find_lat_lon(neighbour_city)
            if neighbour_city_gps == [0, 0]:
                heuristic[neighbour_city] = float(heuristic[city[1]]) - float(road_segments[city[1]][neighbour_city][0])
                distance_goal = float(heuristic[neighbour_city]) + float(city[3])
            else:
                heuristic[neighbour_city] = distance_lat_lon(neighbour_city_gps[0], neighbour_city_gps[1],goal_lat_lon[0], goal_lat_lon[1])
                distance_goal = float(heuristic[neighbour_city]) + float(city[3])
            # longtour implemented by multiplying the distance by -1, turning min priority queue to max prirority queue
            result.append([(-1) * (float(distance_goal)), neighbour_city, city[1],
                           float(city[3]) + float(road_segments[city[1]][neighbour_city][0])])

    return result


#Astar Heuristic:distance between startnode and successor and successor to goal node
#if we dont have gps data about any city/junction we use the parent city's gps and subtract the distance travelled
#  to reach parent
def solve_astar(start_city, end_city,routing_algorithm, cost_function,road_segment):
    parent = {}
    heuristic = {}
    visited_cities = []
    temp = []
    fringe = []
    start_city_gps = find_lat_lon(start_city)
    distance_to_goal = distance_lat_lon(start_city_gps[0],start_city_gps[1],goal_lat_lon[0],goal_lat_lon[1])
    list = [0,start_city,'root',0] #heuristic value,current node,parent,cost of reaching current node from start node
    heuristic[start_city] = 0
    heapq.heappush(fringe,list)
    while len(fringe) > 0:
        current_city = heapq.heappop(fringe)
        if current_city[1] == end_city:
            parent[current_city[1]] = current_city[2]
            return backtrace(parent, start_city, end_city, road_segment, float(current_city[3]))
        for i in temp: #to allow a visited city to revisit if it came by a smaller cost
            if(i[0] == current_city[1] and float(current_city[3]) < float(i[1])):
                if i[0] not  in visited_cities:
                    pass
                else:
                    visited_cities.remove(i[0])
                    parent[current_city[1]] = current_city[2]
                    temp.remove(i)
                    break

        if current_city[1] not  in visited_cities:
            visited_cities.append(current_city[1])
            temp.append([current_city[1],float(current_city[3])])
            parent[current_city[1]] = current_city[2]
            for s in successors_cities_astar(current_city,routing_algorithm,cost_function,road_segment,parent,heuristic):
                if s[1] in visited_cities:
                    continue
                heapq.heappush(fringe,s)
                if s[1] == end_city:
                    parent[s[1]] = current_city[1]
                    return backtrace(parent, start_city, end_city, road_segment, float(current_city[3]))
    return False

#https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
def distance_lat_lon(latitude1, longitude1, latitude2, longitude2):     # Distance in miles between 2 lat lon
    lat1_radians = math.radians(latitude1)
    lat2_radians = math.radians(latitude2)

    diff_lat = math.radians(abs(latitude2 - latitude1))
    diff_lon = math.radians(abs(longitude2 - longitude1))
    a = (math.pow((math.sin(diff_lat / 2.0)), 2)) + math.cos(lat1_radians) * math.cos(lat2_radians) * (
        math.pow((math.sin(diff_lon / 2.0)), 2))
    c = 2.0 * (math.atan2(*(math.sqrt(a), math.sqrt(1 - a))))
    # d = 6371*c  #in km
    distance = 3958.756 * c  # to convert in miles,3958.756 miles is radius of earth
    return distance

#Find lat long of city,if not found return [0,0]
def find_lat_lon(current_city):
    if current_city in city_gps:
        return city_gps[current_city]
    return [0,0]


start_city = sys.argv[1]
end_city =  sys.argv[2]
routing_algorithm = sys.argv[3]

if routing_algorithm == "uniform" or routing_algorithm == "astar" :
    cost_function = sys.argv[4]
else:
    cost_function = "distance"

#Read city gps data
city_gps = read_city_gps()
avg_speed = 0
#Read road_segments data and find average speed on road in US
avg_speed,road_segments_result = read_road_segments()
goal_lat_lon =  find_lat_lon(end_city) #find goal city lat and long

start_time = time.time()
if routing_algorithm == "bfs" or routing_algorithm == "dfs":
    solution = solve_dfs_bfs(start_city,end_city,routing_algorithm,cost_function,road_segments_result)

if routing_algorithm == "uniform":
    solution = solve_uniform(start_city,end_city,routing_algorithm,cost_function,road_segments_result)

if routing_algorithm == "astar":
    solution = solve_astar(start_city,end_city,routing_algorithm,cost_function,road_segments_result)

end_time = time.time()-start_time


print  "Time :" + str(end_time )

#Print the path when you get the solution
if solution != False:
    length,hours,result = solution
    str = ""
    j=0
    for i in result:
        if i == end_city:
            str = str + i
            break
        city1 = i
        city2 =result[j+1]
        distance = road_segments_result[i][result[j+1]][0]
        highway = road_segments_result[i][result[j+1]][2]
        time = round(float(road_segments_result[i][result[j+1]][0])/float(road_segments_result[i][result[j+1]][1]),2)
        print("Go from {0} to {1} via {2} for {3} miles ({4} hours)".format(city1,city2,highway,distance,time))
        str = str + i + " "
        j = j + 1
    print length, hours, str
else:
    print "No Solution"




