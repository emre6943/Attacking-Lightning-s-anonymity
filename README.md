# Attacking Lightning's Anonymity

An attack that enables an intermediary to break the anonymity of the source and destination of a trannsaction in the Lightning network. 

This includes a simulator to simulate transactions using **LND**(https://github.com/lightningnetwork/lnd/blob/master/routing/pathfind.go), **c-Lightning**(https://github.com/ElementsProject/lightning/blob/f3159ec4acd1013427c292038b88071b868ab1ff/common/route.c) and **Eclair**(https://github.com/ACINQ/eclair/blob/master/eclair-core/src/main/scala/fr/acinq/eclair/router/Router.scala). 

All path finding algorithms are based on the Dijkstra's algorithm using a priority queue(https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm).

We modify Eclair to use a generalized version of Dijkstra's algorithm(https://en.wikipedia.org/wiki/K_shortest_path_routing) instead of using Yen's algorithm(https://en.wikipedia.org/wiki/Yen%27s_algorithm). We do not have code to simulate concurrent payments yet. 

The experiment is run on a snapshot of the Lightning Network obtained from https://ln.bigsun.xyz. The set of adversaries is a mixture of nodes with high centrality, low centrality and random nodes. The snapshot as well as the centralities of all nodes are found in data/Snapshot and data/Centrality respectively.

## Code Structure

*populate_graph.py* - Creates a payment graph from a snapshot of the Lightning Network.

*pathFind.py* - Implements Dijkstra and generalized Dijkstra(for 3 best paths) taking the cost function of either LND, c-Lightning or Eclair as argument.

*attack_mixed.py* - Implements an attack where an intermediary finds all potential sources and destinations of a transaction that it is a part of.

*execute.py* - Runs a sample experiment with a set of adversaries on transactions between random pairs of sources and destinations.

*results.py* - Generates statistics and plots from the anonymity sets.

*demo.py* - Simulates a generated network with the given pathing algorithm. Most of the functions are here

## Running an experiment on Snapshot

- Delete everything in results.json and put an empty array "[]"
- Run the SnapShot.py an copy the ADS because they are needed.
- change the ads varaible in NewResults.py to the copied ads.
- Run NewResults.py

Note: If you want to change the routing algorithm or attack find and change it in code. 
Runing the Snapshot takes a long time if you want a fast experiment read bellow.
deanonymize function is important, related to the attack.
random_walk function is related to the pathing.

## Running an experiment on a Generated Network (fast)

- Delete everything in results.json and put an empty array "[]"
- set the if statement to true in demo.py, run demo.py
- copy the Adversaries from demo.py and change the ads varaible in NewResults.py to the copied ads.
- Make sure NewResults.py is using the generated saved network.
- Run NewResults.py

Note: If you want to change the routing algorithm or attack find and change it in code. 
deanonymize function is important, related to the attack.
random_walk function is related to the pathing.
you can change the properties of generated network.


## Metrics

- "Average fee per transaction" is important forthe user and depends on the routing algorithm.Most of the time, the gain in anonymity is paidwith Average fee per transaction. 
- Another price of anonymity is "average number of hops per trans-action", the higher this number is the busier the network, which makes it more prone to attackslike denial of service.
- "Average anonymity source size" is the averagenumber of possible senders per transaction. The bigger this average number the harder to find the real sender, since the sender set is used in generating possible paths.
- "Average anonymity sink size" is the same as Av-erage anonymity source size and it is average number of possible receivers per transaction.
- "Clean any singular ratio" is what the attackers would like to see. It gives the ratio of how manyof the possible senders, or receiver sets had only 1node in it. It just means the attacker knows the sender or the receiver with 100% certainty. This metric is specifically important for the unmodifiedattack on the LND routing, because it isessentially the success rate of that attack. 
- "Cleanall singular ratio" is just like Clean any singularratio, but instead of success defined as knowingthe sender or the receiver, rather success is defined as knowing the sender and the receiver.
- Finally our main metrics to evaluate the attackare defined as "single attack and/or", and "com-bined attack and/or". Single attack is when thecurious nodes do not share information in betweeneach other and guess the sender and the receiverdepending on only their knowledge.  Therefore, "single attack and" is the ratio of correctly guessed sender and receiver,while "single attack or" is when the single attack guessed the sender or the receiver correctly. Com-bined attack is when curious nodes share theirinformation with each other and guess together."Combined attack and" is the ratio of attackers guessing the sender and the receiver correctly, and "combined attack or" is the ratio of attackers guessing the sender or the receiver correctly.



