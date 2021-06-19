from math import inf
from queue import  PriorityQueue
import nested_dict as nd
import random
import matplotlib.pyplot as plt
import networkx as nx
import json
import random as rn

LND_RISK_FACTOR = 0.000000015

def lnd_cost_fun(G, amount, u, v):
    fee = G.edges[v,u]['BaseFee'] + amount * G.edges[v, u]['FeeRate']
    alt = (amount+fee) * G.edges[v, u]["Delay"] * LND_RISK_FACTOR + fee
    return alt

def Dijkstra(G,source,target,amt,cost_function):
    paths = {}
    dist = {}
    delay = {}
    amount =  {}
    # prob = {}
    for node in G.nodes():
        amount[node] = -1
        delay[node] = -1
        dist[node] = inf
    visited = set()
    pq = PriorityQueue()
    dist[target] = 0
    delay[target] = 0
    paths[target] = [target]
    amount[target] = amt
    pq.put((dist[target],target))
    while 0!=pq.qsize():
        curr_cost,curr = pq.get()
        if curr == source:
            return paths[curr],delay[curr],amount[curr],dist[curr]
        if curr_cost > dist[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            if v == source and G.edges[v,curr]["Balance"]>=amount[curr]:
                cost = dist[curr] + amount[curr]*G.edges[v,curr]["Delay"]*LND_RISK_FACTOR
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr]
                    pq.put((dist[v],v))
            if(G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited and v!=source:
                cost = dist[curr] + cost_function(G,amount[curr],curr,v)
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v,curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr] + G.edges[v,curr]["BaseFee"] + amount[curr]*G.edges[v,curr]["FeeRate"]
                    pq.put((dist[v],v))
    return [],-1,-1,-1

def Modified_Dijkstra(G,source,target,amt,cost_function):
    paths = {}
    dist = {}
    delay = {}
    amount =  {}
    # prob = {}
    for node in G.nodes():
        amount[node] = -1
        delay[node] = -1
        dist[node] = inf
    visited = set()
    pq = PriorityQueue()
    dist[target] = 0
    delay[target] = 0
    paths[target] = [target]
    amount[target] = amt
    pq.put((dist[target],target))
    while 0!=pq.qsize():
        curr_cost,curr = pq.get()
        if curr == source:
            return paths, delay, amount, dist, curr
        if curr_cost > dist[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            if v == source and G.edges[v,curr]["Balance"]>=amount[curr]:
                cost = dist[curr] + amount[curr]*G.edges[v,curr]["Delay"]*LND_RISK_FACTOR
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v, curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr]
                    pq.put((dist[v],v))
            if(G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= amount[curr]) and v not in visited and v!=source:
                cost = dist[curr] + cost_function(G,amount[curr],curr,v)
                if cost < dist[v]:
                    dist[v] = cost
                    paths[v] = [v] + paths[curr]
                    delay[v] = G.edges[v,curr]["Delay"] + delay[curr]
                    amount[v] = amount[curr] + G.edges[v,curr]["BaseFee"] + amount[curr]*G.edges[v,curr]["FeeRate"]
                    pq.put((dist[v],v))
    return [],-1,-1,-1,-1

def Random_Walk(G, source, target, amt, cost_function, length):
    # SECURITY bigger lengths more secure transactions
    paths, delays, amounts, dists, first = Modified_Dijkstra(G, source, target, amt, cost_function)
    # no path found
    if first == -1:
        return False, [], -1, -1, -1
    hops = len(paths[first]) - 1
    last_index = 1
    #can make a random hop from any node except the source and sink
    random_nodes = [*range(last_index, hops - 1)]

    added_random_walks = 0

    if len(random_nodes) == 0 and last_index == (hops - 1):
        random_nodes = [last_index]
    left_hops = length - hops

    # Not gonna look for hops
    if hops >= length + 2 or len(random_nodes) == 0:
        # +1 because worst case
        return False, paths[first], delays[first], amounts[first], dists[first]

    # SECURITY also possible with left 1 hops
    while left_hops >= 2 and len(random_nodes) > 0 and added_random_walks < length:
        last_index = random.choice(random_nodes)
        random_nodes.remove(last_index)
        curr = paths[source][last_index]
        walk = []
        for [v, curr] in G.in_edges(curr):
            # SECURITY Am I sure about this ?
            if v not in paths[first][last_index:len(paths[first])] and v != source and G.edges[curr,v]["Balance"] >= amounts[curr]:
                walk.append(v)
        # adding a random walk
        while len(walk) != 0:
            new_road = random.choice(walk)
            walk.remove(new_road)
            if G.edges[new_road, curr]["Balance"] >= amounts[curr]:
                new_path, new_delay, new_amount, new_dist = Dijkstra(G, source, new_road, amt, cost_function)
                #Are we in a dead end, do we have enough hops left?
                if len(new_path) == 0 or (len(new_path) + last_index - len(paths[source])) - 1 >= length:
                    continue
                else:
                    # EFICENCY wana use cost ?
                    paths[new_road] = [new_road] + paths[curr]
                    cost = dists[curr] + amounts[curr] * G.edges[new_road, curr]["Delay"] * LND_RISK_FACTOR
                    dists[new_road] = cost
                    delays[new_road] = G.edges[new_road, curr]["Delay"] + delays[curr]
                    amounts[new_road] = amounts[curr] + G.edges[new_road, curr]["BaseFee"] + amounts[curr] * G.edges[new_road, curr]["FeeRate"]
                    # add the new path
                    for i in range(1, len(new_path) - 1):
                        curr = new_path[::-1][i]
                        pre = new_path[::-1][i - 1]
                        paths[curr] = [curr] + paths[pre]
                        cost = dists[pre] + amounts[pre] * G.edges[curr, pre]["Delay"] * LND_RISK_FACTOR
                        dists[curr] = cost
                        delays[curr] = G.edges[curr, pre]["Delay"] + delays[pre]
                        amounts[curr] = amounts[pre] + G.edges[curr, pre]["BaseFee"] + amounts[pre] * G.edges[curr, pre]["FeeRate"]
                    # ALSO DECREASE THE HOPS
                    paths[source] = new_path + paths[source][last_index:len(paths[source])]
                    delays[source] = G.edges[source, new_path[1]]["Delay"] + delays[new_path[1]]
                    amounts[source] = amounts[new_path[1]]
                    dists[source] = amounts[source] - amt

                    hops = len(paths[source]) - 1
                    random_nodes = [*range(last_index - 1, hops - 1)]
                    left_hops = length - hops
                    added_random_walks += 1
                    break

    # return the new road
    return True, paths[source], delays[source],amounts[source],dists[source]

def Random_Walk_Insertion(G, source, target, amt, cost_function, length):
    # SECURITY bigger lengths more secure transactions
    paths, delays, amounts, dists, first = Modified_Dijkstra(G, source, target, amt, cost_function)
    # no path found
    if first == -1:
        return False, [], -1, -1, -1
    hops = len(paths[first]) - 1
    random_nodes = [*range(1, hops)]

    # Not gonna look for hops
    if length <= 0 or len(random_nodes) == 0:
        return False, paths[first], delays[first], amounts[first], dists[first]

    # SECURITY also possible with left 1 hops
    while len(random_nodes) > 0:
        last_index = random.choice(random_nodes)
        random_nodes.remove(last_index)
        random_branch_point = paths[source][last_index]
        curr = random_branch_point
        random_length = 0

        #Random Walk
        while random_length < length and len(random_nodes) > 0:
            walk = []
            for [v, curr] in G.in_edges(curr):
                # if the going node can carry the amount and it is not the source and the sink
                if v != target and v != source and G.edges[curr,v]["Balance"] >= amounts[curr]:
                    walk.append(v)
            if len(walk) <= 0:
                break
            # Random Walk
            new_road = random.choice(walk)
            paths[new_road] = [new_road] + paths[curr]
            cost = dists[curr] + amounts[curr] * G.edges[new_road, curr]["Delay"] * LND_RISK_FACTOR
            dists[new_road] = cost
            delays[new_road] = G.edges[new_road, curr]["Delay"] + delays[curr]
            amounts[new_road] = amounts[curr] + G.edges[new_road, curr]["BaseFee"] + amounts[curr] * \
                                G.edges[new_road, curr]["FeeRate"]
            random_length += 1
            curr = new_road

        #Is this random walk possible
        new_path, new_delay, new_amount, new_dist = Dijkstra(G, source, curr, amounts[curr], cost_function)
        if len(new_path) != 0:
            for i in range(1, len(new_path) - 1):
                curr = new_path[::-1][i]
                pre = new_path[::-1][i - 1]
                paths[curr] = [curr] + paths[pre][1::]
                cost = dists[pre] + amounts[pre] * G.edges[curr, pre]["Delay"] * LND_RISK_FACTOR
                dists[curr] = cost
                delays[curr] = G.edges[curr, pre]["Delay"] + delays[pre]
                amounts[curr] = amounts[pre] + G.edges[curr, pre]["BaseFee"] + amounts[pre] * G.edges[curr, pre]["FeeRate"]

            paths[source] = new_path + paths[curr][1::]
            delays[source] = G.edges[source, new_path[1]]["Delay"] + delays[new_path[1]]
            amounts[source] = amounts[new_path[1]]
            dists[source] = amounts[source] - amt
            return True, paths[source], delays[source], amounts[source], dists[source]

    # return the new road
    return False, paths[source], delays[source],amounts[source],dists[source]

def Weighted_Random_Walk_Insertion(G, source, target, amt, cost_function, length):
    # SECURITY bigger lengths more secure transactions
    paths, delays, amounts, dists, first = Modified_Dijkstra(G, source, target, amt, cost_function)
    # no path found
    if first == -1:
        return False, [], -1, -1, -1
    hops = len(paths[first]) - 1
    random_nodes = [*range(1, hops)]

    # Not gonna look for hops
    if length <= 0 or len(random_nodes) == 0:
        return False, paths[first], delays[first], amounts[first], dists[first]

    #weights
    weights = {}
    total = 0
    for num in random_nodes:
        weights[num] = len(G.in_edges(paths[source][num]))
        total += len(G.in_edges(paths[source][num]))
    for i in range(len(weights)):
        weights[random_nodes[i]] = weights[random_nodes[i]]/total

    # SECURITY also possible with left 1 hops
    while len(random_nodes) > 0:
        last_index = random.choices(random_nodes, weights=list(weights.values()), k=1)[0]
        random_nodes.remove(last_index)
        weights.pop(last_index)

        random_branch_point = paths[source][last_index]
        curr = random_branch_point
        random_length = 0

        #Random Walk
        while random_length < length and len(random_nodes) > 0:
            walk = []
            walk_weights = []
            w_total = 0
            for [v, curr] in G.in_edges(curr):
                # if the going node can carry the amount and it is not the source and the sink
                if v != target and v != source and G.edges[curr,v]["Balance"] >= amounts[curr]:
                    walk.append(v)
                    walk_weights.append(len(G.in_edges(v)))
                    w_total += len(G.in_edges(v))

            if len(walk) <= 0:
                break

            for i in range(len(walk_weights)):
                walk_weights[i] = walk_weights[i] / w_total

            # Random Walk
            new_road = random.choices(walk, weights=walk_weights, k=1)[0]
            paths[new_road] = [new_road] + paths[curr]
            cost = dists[curr] + amounts[curr] * G.edges[new_road, curr]["Delay"] * LND_RISK_FACTOR
            dists[new_road] = cost
            delays[new_road] = G.edges[new_road, curr]["Delay"] + delays[curr]
            amounts[new_road] = amounts[curr] + G.edges[new_road, curr]["BaseFee"] + amounts[curr] * \
                                G.edges[new_road, curr]["FeeRate"]
            random_length += 1
            curr = new_road

        #Is this random walk possible
        new_path, new_delay, new_amount, new_dist = Dijkstra(G, source, curr, amounts[curr], cost_function)
        if len(new_path) != 0:
            for i in range(1, len(new_path) - 1):
                curr = new_path[::-1][i]
                pre = new_path[::-1][i - 1]
                paths[curr] = [curr] + paths[pre][1::]
                cost = dists[pre] + amounts[pre] * G.edges[curr, pre]["Delay"] * LND_RISK_FACTOR
                dists[curr] = cost
                delays[curr] = G.edges[curr, pre]["Delay"] + delays[pre]
                amounts[curr] = amounts[pre] + G.edges[curr, pre]["BaseFee"] + amounts[pre] * G.edges[curr, pre]["FeeRate"]

            paths[source] = new_path + paths[curr][1::]
            delays[source] = G.edges[source, new_path[1]]["Delay"] + delays[new_path[1]]
            amounts[source] = amounts[new_path[1]]
            dists[source] = amounts[source] - amt
            return True, paths[source], delays[source], amounts[source], dists[source]

    # return the new road
    return False, paths[source], delays[source],amounts[source],dists[source]

def dest_reveal_new(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    flag = True

    while(flag):
        level+=1
        if(level == 4):
            flag1 = False
            break
        t1 = T[level - 1]["nodes"]
        d1 = T[level - 1]["delays"]
        v1 = T[level - 1]["visited"]
        a1 = T[level - 1]["amounts"]
        t2 = []
        d2 = []
        p2 = []
        v2 = [[]]
        a2 = []
        for i in range(0,len(t1)):
            u = t1[i]
            for [u,v] in G.out_edges(u):
                if(v!=pre and v!=adversary  and v!=next and v not in v1[i] and (d1[i] - G.edges[u,v]["Delay"])>=0 and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
        T[level]["nodes"] = t2
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        if(len(t2) == 0):
            flag = False
    level = level - 1
    while(level>=0):
        t = T[level]["nodes"]
        d = T[level]["delays"]
        p = T[level]["previous"]
        a = T[level]["amounts"]
        v = T[level]["visited"]
        for i in range(0, len(t)):
            if(d[i] == 0):
                path = []
                level1 = level
                path.append(T[level1]["nodes"][i])
                loc = T[level1]["previous"][i]
                while (level1 > 0):
                    level1 = level1 - 1
                    path.append(T[level1]["nodes"][loc])
                    loc = T[level1]["previous"][loc]
                path.reverse()
                path = [pre,adversary]+path
                if (len(path) == len(set(path))):
                    amt = a[i]
                    pot = path[len(path) - 1]
                    sources = deanonymize(G,pot,path,amt,lnd_cost_fun)
                    if sources != None:
                        anon_sets[pot] = list(sources)
        level = level - 1
    return anon_sets,flag1

def dest_reveal_new_changed(G,adversary,delay,amount,pre,next):
    T = nd.nested_dict()
    flag1 = True
    anon_sets = nd.nested_dict()
    level = 0
    T[0]["nodes"] = [next]
    T[0]["delays"] = [delay]
    T[0]["previous"] = [-1]
    T[0]["visited"] = [[pre,adversary,next]]
    T[0]["amounts"] = [amount]
    flag = True

    while(flag):
        level+=1
        if(level == 4):
            flag1 = False
            break
        t1 = T[level - 1]["nodes"]
        d1 = T[level - 1]["delays"]
        v1 = T[level - 1]["visited"]
        a1 = T[level - 1]["amounts"]
        t2 = []
        d2 = []
        p2 = []
        v2 = [[]]
        a2 = []
        for i in range(0,len(t1)):
            u = t1[i]
            for [u,v] in G.out_edges(u):
                # it can be the next removed 'and v!=next'
                # can be a visited node no? removed 'and v not in v1[i]'
                if(v!=pre and v!=adversary and (d1[i] - G.edges[u,v]["Delay"])>=0 and (G.edges[u,v]["Balance"]+G.edges[v,u]["Balance"])>=((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"]))):
                    t2.append(v)
                    d2.append(d1[i] - G.edges[u,v]["Delay"])
                    p2.append(i)
                    v2.append(v1[i]+[v])
                    a2.append(((a1[i] - G.edges[u, v]["BaseFee"]) / (1 + G.edges[u, v]["FeeRate"])))
        T[level]["nodes"] = t2
        T[level]["delays"] = d2
        T[level]["previous"] = p2
        T[level]["visited"] = v2
        T[level]["amounts"] = a2
        if(len(t2) == 0):
            flag = False
    level = level - 1
    while(level>=0):
        t = T[level]["nodes"]
        d = T[level]["delays"]
        p = T[level]["previous"]
        a = T[level]["amounts"]
        v = T[level]["visited"]
        for i in range(0, len(t)):
            if(d[i] == 0):
                path = []
                level1 = level
                path.append(T[level1]["nodes"][i])
                loc = T[level1]["previous"][i]
                while (level1 > 0):
                    level1 = level1 - 1
                    path.append(T[level1]["nodes"][loc])
                    loc = T[level1]["previous"][loc]
                path.reverse()
                path = [pre,adversary]+path
                if (len(path) == len(set(path))):
                    amt = abs(a[i])
                    pot = path[len(path) - 1]
                    sources = deanonymize_changed(G,pot,path,amt,lnd_cost_fun)
                    if sources != None:
                        anon_sets[pot] = list(sources)
        level = level - 1
    return anon_sets,flag1

def deanonymize_changed (G,target,path,amt,cost_function):
    # print("path p ", path)
    # print("target ", target)
    # print("amt ", amt)
    pq = PriorityQueue()
    delays = {}
    costs = {}
    paths = nd.nested_dict()
    paths1 = nd.nested_dict()
    dists = {}
    visited = set()
    previous = {}
    done = {}
    prob = {}
    sources = []
    pre = path[0]
    adv = path[1]
    nxt = path[2]
    for node in G.nodes():
        previous[node] = -1
        delays[node] = -1
        costs[node] = inf
        paths[node] = []
        dists[node] = inf
        done[node] = 0
        paths1[node] = []
        prob[node] = 1
    dists[target] = 0
    paths[target] = [target]
    costs[target] = amt
    delays[target] = 0
    pq.put((dists[target],target))
    flag1 = 0
    flag2 = 0
    while(0!=pq.qsize()):
        curr_cost,curr = pq.get()
        if curr_cost > dists[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            # bad assumption removed "and v not in visited"
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= costs[curr]) :
                #what is this assumption
                if done[v] == 0:
                    paths1[v] = [v]+paths[curr]
                    done[v] = 1
                cost = dists[curr]+ cost_function(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    pq.put((dists[v],v))
        if(curr in path[1:]):
            ind = path.index(curr)
            # if(paths[curr]!=path[ind:]):
            #     return None
            if curr == adv:
                flag1 = 1
        if(curr == pre):
            # if paths[pre] != path:
            #     return [pre]
            # else:
            sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            if pre in paths[curr]:
                for [v,curr] in G.in_edges(curr):
                        if v not in paths[curr]:
                            sources.append(v)
    sources = set(sources)
    return sources

def deanonymize(G,target,path,amt,cost_function):
    pq = PriorityQueue()
    delays = {}
    costs = {}
    paths = nd.nested_dict()
    paths1 = nd.nested_dict()
    dists = {}
    visited = set()
    previous = {}
    done = {}
    prob = {}
    sources = []
    pre = path[0]
    adv = path[1]
    nxt = path[2]
    for node in G.nodes():
        previous[node] = -1
        delays[node] = -1
        costs[node] = inf
        paths[node] = []
        dists[node] = inf
        done[node] = 0
        paths1[node] = []
        prob[node] = 1
    dists[target] = 0
    paths[target] = [target]
    costs[target] = amt
    delays[target] = 0
    pq.put((dists[target],target))
    flag1 = 0
    flag2 = 0
    while(0!=pq.qsize()):
        curr_cost,curr = pq.get()
        if curr_cost > dists[curr]:
            continue
        visited.add(curr)
        for [v,curr] in G.in_edges(curr):
            if (G.edges[v, curr]["Balance"] + G.edges[curr, v]["Balance"] >= costs[curr]) and v not in visited:
                if done[v] == 0:
                    paths1[v] = [v]+paths[curr]
                    done[v] = 1
                cost = dists[curr]+ cost_function(G,costs[curr],curr,v)
                if cost < dists[v]:
                    paths[v] = [v]+paths[curr]
                    dists[v] = cost
                    delays[v] = delays[curr] + G.edges[v,curr]["Delay"]
                    costs[v] = costs[curr] + G.edges[v, curr]["BaseFee"] + costs[curr] * G.edges[v, curr]["FeeRate"]
                    pq.put((dists[v],v))
        if(curr in path[1:]):
            ind = path.index(curr)
            if(paths[curr]!=path[ind:]):
                return None
            if curr == adv:
                flag1 = 1
        if(curr == pre):
            if paths[pre] != path:
                return [pre]
            else:
                sources.append(pre)
            flag2 = 1
        if flag1 == 1 and flag2 == 1:
            if pre in paths[curr]:
                for [v,curr] in G.in_edges(curr):
                        if v not in paths[curr]:
                            sources.append(v)
    sources = set(sources)
    return sources

def route(G,path,delay,amt,ads,amt1,file):
    G1 = G.copy()
    cost = amt
    comp_attack = []
    anon_sets =[]
    attacked = 0
    G.edges[path[0],path[1]]["Balance"] -= amt
    G.edges[path[1],path[0]]["Locked"] = amt
    delay = delay - G.edges[path[0],path[1]]["Delay"]
    i = 1
    if len(path) == 2:
        G.edges[path[1],path[0]]["Balance"] += G.edges[path[1],path[0]]["Locked"]
        G.edges[path[1], path[0]]["Locked"] = 0
        transaction = {"sender": path[0], "recipient": path[1], "path" : path, "delay": delay, "amount":amt1,
                       "Cost": cost, "attacked":0,
                    "success":True,"anon_sets":anon_sets,"comp_attack":comp_attack}
        transactions.append(transaction)
        return True
    while(i < len(path)-1):
        amt = (amt - G.edges[path[i], path[i+1]]["BaseFee"]) / (1 + G.edges[path[i], path[i+1]]["FeeRate"])
        if path[i] in ads:
            attacked+=1
            dests = []
            delay1 = delay - G.edges[path[i],path[i+1]]["Delay"]
            # the dest reveal is the one that builds the possible sender recievers
            B,flag = dest_reveal_new(G1,path[i],delay1,amt,path[i-1],path[i+1])
            # B, flag = dest_reveal_new_changed(G1, path[i], delay1, amt, path[i - 1], path[i + 1])
            for j in B:
                dest = {j:B[j]}
                dests.append(dest)
            if flag == True:
                comp_attack.append(1)
            else:
                comp_attack.append(0)
            anon_set = {path[i]:dests}
            anon_sets.append(anon_set)
        if(G.edges[path[i],path[i+1]]["Balance"] >= amt):
            G.edges[path[i], path[i+1]]["Balance"] -= amt
            G.edges[path[i+1], path[i]]["Locked"] = amt
            if i == len(path) - 2:
                G.edges[path[i+1],path[i]]["Balance"] += G.edges[path[i+1], path[i]]["Locked"]
                G.edges[path[i+1], path[i]]["Locked"] = 0
                j = i - 1
                while j >= 0:
                    G.edges[path[j + 1], path[j]]["Balance"] += G.edges[path[j + 1], path[j]]["Locked"]
                    G.edges[path[j + 1], path[j]]["Locked"] = 0
                    j = j-1
                    transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                                   "amount": amt1, "Cost": cost, "attacked": attacked,
                                   "success": True, "anon_sets": anon_sets, "comp_attack": comp_attack}
                    transactions.append(transaction)
                return True
            delay = delay - G.edges[path[i],path[i+1]]["Delay"]
            i += 1
        else:
            j = i - 1
            while j >= 0:
                G.edges[path[j],path[j+1]]["Balance"] += G.edges[path[j+1],path[j]]["Locked"]
                G.edges[path[j + 1], path[j]]["Locked"] = 0
                j = j-1
                transaction = {"sender": path[0], "recipient": path[len(path)-1], "path": path, "delay": delay,
                               "amount": amt1, "Cost": cost, "attacked": attacked,
                               "success": False, "anon_sets": anon_sets, "comp_attack": comp_attack}
                transactions.append(transaction)
            return False
# run it
if False:
    G = nx.barabasi_albert_graph(500,10,65)
    G = nx.DiGraph(G)

    for [u,v] in G.edges():
        G.edges[u,v]["Delay"] = 10 * rn.randint(1,10)
        G.edges[u,v]["BaseFee"] = 0.1 * rn.randint(1,10)
        G.edges[u,v]["FeeRate"] = 0.0001 * rn.randint(1,10)
        G.edges[u,v]["Balance"] = rn.randint(100,10000)

    transactions = []

    B = nx.betweenness_centrality(G)

    ads = []
    for i in range(0,50):
        node = -1
        max = -1
        for u in G.nodes():
            if B[u] >= max and u not in ads:
                max = B[u]
                node = u
        if node not in ads:
            ads.append(node)

    print("Adversaries:",ads)

    nx.draw(G)
    plt.show()
    nx.write_gexf(G, "network.gexf")

    i=0
    file = "results.json"
    perc = 0
    tot = 0
    while (i < 100):
        u = -1
        v = -1
        while (u == v or (u not in G.nodes()) or (v not in G.nodes())):
            u = rn.randint(0, 99)
            v = rn.randint(0, 99)
        amt = 0
        if (i % 3 == 0):
            amt = rn.randint(1, 10)
        elif (i % 3 == 1):
            amt = rn.randint(10, 100)
        elif (i % 3 == 2):
            amt = rn.randint(100, 1000)
        print("AMT -> " + str(amt))
        rando = False
        # path, delay, amount, dist = Dijkstra(G, u, v, amt, lnd_cost_fun)
        # print("path -> " + str(path) + ", delay -> " + str(delay) + ", amount -> " + str(amount) + ", dist -> " + str(dist))

        # rando, path, delay, amount, dist = Random_Walk(G, u, v, amt, lnd_cost_fun, 4)
        # print(rando, "path -> " + str(path) + ", delay -> " + str(delay) + ", amount -> " + str(amount) + ", dist -> " + str(dist))
        #
        # rando, path, delay, amount, dist = Random_Walk_Insertion(G, u, v, amt, lnd_cost_fun, 4)
        # print(rando, "path -> " + str(path) + ", delay -> " + str(delay) + ", amount -> " + str(amount) + ", dist -> " + str(dist))

        rando, path, delay, amount, dist = Weighted_Random_Walk_Insertion(G, u, v, amt, lnd_cost_fun, 4)
        print(rando, "path -> " + str(path) + ", delay -> " + str(delay) + ", amount -> " + str(amount) + ", dist -> " + str(dist))

        if (len(path) > 0):
            T = route(G, path, delay, amount, ads, amt, file)
            if rando:
                perc += 1
            tot += 1
            print( "---", T, "---")
        if len(path) > 2:
            print(i,path, "done")
            i += 1
    with open(file,'r') as json_file:
        data = json.load(json_file)
    data.append(transactions)
    with open(file,'w') as json_file:
        json.dump(data,json_file,indent=1)

    print("transactions length ", len(transactions))
    print(f'total random transactions percentage {round((perc/tot) * 100, 2)}%')

