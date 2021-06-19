import csv
import ast
import nested_dict as nd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx
import populate_graph as pg
from math import inf
from queue import  PriorityQueue
import random
import json

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
ads = [2634, 8075, 5347, 1083, 5093, 4326, 4126, 2836, 5361, 10572, 5389, 3599, 9819, 4828, 3474, 8808, 93, 9530, 9515,
       2163]
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

avg_anon_sink_size = avg_anon_sink_size / num_attacks
avg_anon_source_size = avg_anon_source_size / (avg_anon_sink_size * num_attacks)
print(f'Avarage anonimity source size: {avg_anon_source_size}')
print(f'Avarage anonimity sink size: {avg_anon_sink_size}')

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


