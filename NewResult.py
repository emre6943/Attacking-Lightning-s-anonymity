import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import utils as ut
import populate_graph as pg
from math import inf
from queue import  PriorityQueue
import random
import json
import pickle

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
        if  int(curr) == int(source):
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

file = "results.json"

# Open the results file and transfer the in an array
results = []
with open(file, 'r') as json_file:
    results_json = json.load(json_file)
results.append(results_json)

path = []

# Number of Transactions
num_transactions = 0

# Number of transactions attacked
num_attacked = 0

# Total number of attack instances
num_attacks = 0

# Array storing the number of recipients for each attack instance, followed by those that that had phase I completed and not respectively
dest_count = []
dest_count_comp = []
dest_count_incomp = []

# Arrays storing the number of senders for each attack instance, followed by those that that had phase I completed and not respectively
source_count = []
source_count_comp = []
source_count_incomp = []

# Arrays storing the distances of the recipient and the sender from the adversary respectively
dist_dest = []
dist_source = []

# Number of attack instances in which the sender and recipient pair was successfully found
pair_found = 0

# Number of attack instances that completed phase I
num_comp = 0

# Number of attack instances for which the size of the recipient set was 1 and similarly for the sender
sing_dest = 0
sing_source = 0

# Number of attack instances having both the sender and recipient sets singular
sing_all = 0

# Number of attack instances having atleast one of the sender and recipient sets singular
sing_any = 0
# changed with 97 randoms + good bois

ads = [2634, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515, 2163, 2818, 8435, 10737, 9826, 9411, 9426, 8581, 3954, 7142, 8713, 819, 7679, 1530, 800, 9508, 4008, 569, 3859, 2354, 1478, 8844, 7597, 9453, 8917, 5838, 10324, 5312, 5812, 3172, 820, 5737, 1878, 3190, 3648, 9093, 4854, 8879, 8982, 4521, 2304, 789, 5600, 10914, 10832, 11082, 4221, 1184, 2815, 2997, 2804, 6711, 2076, 3923, 6985, 6625, 10046, 3679, 9054, 24, 867, 2632, 4560, 3134, 8332, 1636, 8186, 8562, 4131, 257, 999, 1179, 9818, 9750, 10584, 7787, 5908]

# ads =  [10, 13, 14, 9, 12, 16, 17, 11, 1, 18, 35, 23, 20, 28, 19, 26, 25, 24, 74, 22, 15, 27, 37, 21, 47, 63, 3, 51, 30, 73, 31, 78, 52, 57, 29, 50, 4, 48, 53, 33, 103, 64, 60, 70, 34, 44, 40, 54, 65, 85]
# ads = [10, 13, 14, 9, 12, 16, 17, 11, 1, 18]
# ads = [2634, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515,
#        2163]
# ads = [3, 2, 7, 25, 5, 1, 11, 10, 6, 15]

#false positive
false_positive_or = 0
false_positive_and = 0

# total amount of hops
hops = 0

# total amount of fee paid
fee = 0

# anon sizes
avg_anon_source_size = 0
avg_anon_sink_size = 0

# omgomg
omg = 0
omg_yan = 0
omgomg = 0
omgomg_yan = 0

# the graph
# G = nx.read_gexf("network.gexf")
G = nx.read_gexf("snapshot.gexf")

# populate the graph from the snapshot

# G1 = nx.DiGraph()
# G1, m = pg.populate_nodes(G1)
# G1, m1 = pg.populate_channels(G1, m, ut.getBlockHeight())
# G1 = pg.populate_policies(G1, m1)
#
# # curate nodes and channels removing channels that are closed and those that do not have public policies
# G = nx.DiGraph()
# for [u, v] in G1.edges():
#     if (G1.edges[u, v]["marked"] == 1 and G1.edges[v, u]["marked"] == 1):
#         if (u not in G.nodes()):
#             G.add_node(u)
#             G.nodes[u]["name"] = G1.nodes[u]["name"]
#             G.nodes[u]["pubadd"] = G1.nodes[u]["pubadd"]
#             G.nodes[u]["Tech"] = G1.nodes[u]["Tech"]
#             # print(G1.nodes[u]["Tech"])
#         if (v not in G.nodes()):
#             G.add_node(v)
#             G.nodes[v]["name"] = G1.nodes[v]["name"]
#             G.nodes[v]["pubadd"] = G1.nodes[v]["pubadd"]
#             G.nodes[v]["Tech"] = G1.nodes[v]["Tech"]
#             # print(G1.nodes[v]["Tech"])
#         G.add_edge(u, v)
#         G.edges[u, v]["Balance"] = G1.edges[u, v]["Balance"]
#         G.edges[u, v]["Age"] = G1.edges[u, v]["Age"]
#         G.edges[u, v]["BaseFee"] = G1.edges[u, v]["BaseFee"]
#         G.edges[u, v]["FeeRate"] = G1.edges[u, v]["FeeRate"]
#         G.edges[u, v]["Delay"] = G1.edges[u, v]["Delay"]
#         G.edges[u, v]["id"] = G1.edges[u, v]["id"]

nx.draw(G)
plt.show()


# Dictionary for storing the number of attack instances of each adversary
ad_attacks = {}
for ad in ads:
    ad_attacks[ad] = 0

# Go over the results and update each of the above variables for each attack instance
for i in results:
    for j in i:
        for k in j:
            if k["path"] != path:
                path = k["path"]
                fee += k["Cost"] - k["amount"]
                hops += (len(path) - 1)
                num_transactions += 1
                print(num_transactions)
                if k["attacked"] > 0:
                    num_attacked += 1
                    anon_sets = k["anon_sets"]
                    the_reversing_paths = []
                    the_reversing_weights = []
                    for ad in anon_sets:
                        num_attacks += 1
                        num = -1
                        for adv in ad:
                            sources = []
                            sinks = []
                            ad_attacks[int(adv)] += 1
                            num += 1
                            avg_anon_sink_size += len(ad[adv])
                            for dest in ad[adv]:
                                for rec in dest:
                                    sinks.append(rec)
                                    avg_anon_source_size += len(dest[rec])

                                    if int(rec) == k["recipient"] and k["sender"] in dest[rec]:
                                        pair_found += 1
                                    for tech in dest[rec]:
                                        sources.append(tech)
                                        # for s in dest[rec][tech]:
                                        #    sources.append(s)
                            if len(set(sources)) > 0:
                                ind = k["path"].index(int(adv))
                                dist_dest.append(len(k["path"]) - 1 - ind)
                                dist_source.append(ind)
                                if (k["comp_attack"][num] == 1):
                                    dest_count_comp.append(len(ad[adv]))
                                    num_comp += 1
                                else:
                                    dest_count_incomp.append(len(ad[adv]))
                                dest_count.append(len(ad[adv]))
                                if (len(ad[adv]) == 1):
                                    sing_dest += 1
                                if (k["comp_attack"][num] == 1):
                                    source_count_comp.append(len(set(sources)))
                                else:
                                    source_count_incomp.append(len(set(sources)))
                                source_count.append(len(set(sources)))
                                if (len(set(sources)) == 1):
                                    sing_source += 1
                                if (len(ad[adv]) == 1) or (len(set(sources)) == 1):
                                    sing_any += 1
                                    if (len(ad[adv]) == 1) and k["recipient"] not in list(ad[adv][0].keys()):
                                        false_positive_or += 1
                                    elif (len(set(sources)) == 1) and sources[0] != k["sender"]:
                                        false_positive_or += 1
                                if (len(ad[adv]) == 1) and (len(set(sources)) == 1):
                                    sing_all += 1
                                    if k["recipient"] not in list(ad[adv][0].keys()) and sources[0] != k["sender"]:
                                        false_positive_and += 1

                            print("reversing")
                            #reverse the path
                            paths = []
                            lens = []
                            print("sources len ", len(sources))
                            print("sinks len ", len(sinks))
                            print("changed")
                            if len(sources) > 1000:
                                sources = sources[:500:]
                            if len(sinks) > 1000:
                                sinks = sinks[:500:]
                            print("sources len ", len(sources))
                            print("sinks len ", len(sinks))
                            for s in sources:
                                for t in sinks:
                                    for i in range(10):
                                        rando, pp, delay, amount, dist = Weighted_Random_Walk_Insertion(G, str(s), str(t), k["amount"],
                                                                                                      lnd_cost_fun, 4)
                                        # pp, delay, amount, dist = Dijkstra(G, s, t, k["amount"], lnd_cost_fun)
                                        if adv in pp:
                                            if pp in paths:
                                                index = paths.index(pp)
                                                lens[index] += 1
                                                index = the_reversing_paths.index(pp)
                                                the_reversing_weights[index] += 1
                                            else:
                                                paths.append(pp)
                                                lens.append(1)
                                                the_reversing_paths.append(pp)
                                                the_reversing_weights.append(1)

                            if len(paths) > 0:
                                max_le = 0
                                index = -1
                                for i in range(len(lens)):
                                    if lens[i] > max_le:
                                        max_le = lens[i]
                                        index = i

                                p = paths[index]
                                if str(k["sender"]) == p[0] and str(k["recipient"]) == p[len(p) - 1]:
                                    omg += 1
                                elif str(k["sender"]) == p[0] or str(k["recipient"]) == p[len(p) - 1]:
                                    omg_yan += 1
                            print("single attack done")

                    #revcerse with all the ads
                    middle = []
                    for ad in anon_sets:
                        if list(ad.keys())[0] not in middle:
                            middle.append(list(ad.keys())[0])

                    ppaths = []
                    wweights = []
                    max_w = 0
                    index = -1
                    print("paths ", the_reversing_paths)
                    if len(the_reversing_paths) > 0:
                        for i in range(len(the_reversing_paths)):
                            condition = True
                            for m in middle:
                                if m not in the_reversing_paths[i]:
                                    condition = False
                            if condition:
                                ppaths.append(the_reversing_paths[i])
                                wweights.append(the_reversing_weights[i])

                        for i in range(len(wweights)):
                            if wweights[i] > max_w:
                                max = wweights[i]
                                index = i
                        print("cleaned ", ppaths)
                        if index != -1:
                            p = ppaths[index]
                            if str(k["sender"]) == p[0] and str(k["recipient"]) == p[len(p) - 1]:
                                omgomg += 1
                            elif str(k["sender"]) == p[0] or str(k["recipient"]) == p[len(p) - 1]:
                                omgomg_yan += 1

                    print("done")





# Print the metrics
def perc(num):
    return round(num * 100, 2)


# Print the metrics
print(f'Transactions: {num_transactions}')
print(f'Transactions attacked: {num_attacked}')
print(f'Attacks: {num_attacks}')
print(f'Pairs found: {pair_found}')
print(f'Sources found per attack: {source_count}')
print(f'Destinations found per attack: {dest_count}')

attack_transaction_ratio = num_attacked / num_transactions if num_transactions != 0 else 0
attack_attacked_ratio = num_attacks / num_attacked if num_attacked != 0 else 0
print(f'Attacked/Transactions ratio: {attack_transaction_ratio} ({perc(attack_transaction_ratio)}%)')
print(f'Attacks/Attacked ratio: {attack_attacked_ratio} ({perc(attack_attacked_ratio)}%)')
print('Correlation destination to distance\n', np.corrcoef(dest_count, dist_dest))
print('Correlation source to distance\n', np.corrcoef(source_count, dist_source))

sing_source_ratio = sing_source / num_attacks if num_attacks != 0 else 0
sing_dest_ratio = sing_dest / num_attacks if num_attacks != 0 else 0
sing_any_ratio = sing_any / num_attacks if num_attacks != 0 else 0
sing_all_ratio = sing_all / num_attacks if num_attacks != 0 else 0
complete_one_attack_ratio = num_comp / num_attacks if num_attacks != 0 else 0

false_ratio_or = false_positive_or / sing_any if sing_any != 0 else 0
false_ratio_and = false_positive_and / sing_all if sing_all != 0 else 0

clean_any = (sing_any - false_positive_or)/num_attacks if num_attacks != 0 else 0
clean_all = (sing_all - false_positive_and)/num_attacks if num_attacks != 0 else 0

print(f'Singular sources ratio: {sing_source_ratio} ({perc(sing_source_ratio)}%)')
print(f'Singular destination ratio: {sing_dest_ratio} ({perc(sing_dest_ratio)}%)')
print(f'Singular source or destination ratio: {sing_any_ratio} ({perc(sing_any_ratio)}%)')
print(f'Singular source or destination false positive ratio: {false_ratio_or} ({perc(false_ratio_or)}%)')
print(f'Clean any singular ratio: {clean_any} ({perc(clean_any)}%)')
print(f'Both Singular ratio: {sing_all_ratio} ({perc(sing_all_ratio)}%)')
print(f'Both Singular ratio false positive: {false_ratio_and} ({perc(false_ratio_and)}%)')
print(f'Clean all singular ratio: {clean_all} ({perc(clean_all)}%)')
print(f'Complete I phase ratio: {complete_one_attack_ratio} ({perc(complete_one_attack_ratio)}%)')

avg_hops = hops / num_transactions if num_transactions != 0 else 0
avg_fee = fee / num_transactions if num_transactions != 0 else 0
pair_found_ratio = pair_found / num_attacks if num_attacks != 0 else 0
print(f'Avarage number of hops per transaction: {avg_hops}')
print(f'Avarage fee per transaction: {avg_fee}')
print(f'Pairs found ration : {pair_found_ratio}%')

avg_anon_sink_size = avg_anon_sink_size / num_attacks if num_attacks != 0 else 0
avg_anon_source_size = avg_anon_source_size / (avg_anon_sink_size * num_attacks) if num_attacks != 0 or avg_anon_sink_size else 0
print(f'Avarage anonimity source size: {avg_anon_source_size}')
print(f'Avarage anonimity sink size: {avg_anon_sink_size}')

# the new revers engineering attack
single = omg / num_attacks if num_attacks != 0 else 0
combined = omgomg / num_transactions if num_transactions != 0 else 0
single_or = omg_yan / num_attacks if num_attacks != 0 else 0
combined_or = omgomg_yan / num_transactions if num_transactions != 0 else 0
print(f'Single attack: {single}, ({perc(single)}%)')
print(f'Combined attack: {combined}, ({perc(combined)}%)')
print(f'Single attack or: {single_or}, ({perc(single_or)}%)')
print(f'Combined attack or: {combined_or}, ({perc(combined_or)}%)')


# Plot the sender and recipient anonymity sets respectively
plot1 = sns.ecdfplot(data=dest_count_comp, legend='Phase I complete', marker='|', linewidth=1.5, linestyle=':')
plot2 = sns.ecdfplot(data=dest_count_incomp, legend='Phase I incomplete', marker='|', linewidth=1.5, linestyle=':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Phase I complete', 'Phase I incomplete'), scatterpoints=1, loc='lower right', ncol=1, fontsize=16)
plt.xlabel("Size of anonymity set")

plt.ylabel("CDF")
plt.show()

plot1 = sns.ecdfplot(data=source_count_comp, legend='Phase I complete', marker='|', linewidth=1.5, linestyle=':')
plot2 = sns.ecdfplot(data=source_count_incomp, legend='Phase I incomplete', marker='|', linewidth=1.5, linestyle=':')
plot1.set(xscale='log')
plot2.set(xscale='log')
plt.legend(('Phase I complete', 'Phase I incomplete'), scatterpoints=1, loc='lower right', ncol=1, fontsize=16)
plt.xlabel("Size of anonymity set")
plt.ylabel("CDF")
plt.show()


