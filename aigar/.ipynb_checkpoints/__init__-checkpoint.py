from gym.envs.registration import register

# Define all possible options:
greedy_opts = [0, 1, 2, 5]
rgb_opts = [True, False]
split_opts = [False, True]
eject_opts = [False, True]
# Register envs:
name = 'Aigar'
for greedy in greedy_opts:
    for rgb in rgb_opts:
        for split in split_opts:
            for eject in eject_opts:
                if eject and not split:
                    continue
                
                if greedy:
                    new_name = name + "Greedy" + str(greedy)
                else:
                    new_name = name + "Pellet"
                if not rgb:
                    new_name += "Grid"
                if split:
                    new_name += "Split"
                if eject:
                    new_name += "Eject"
                new_name += "-v0"
                register(
                    id = new_name,
                    entry_point = 'aigar.envs:AigarEnv',
                    kwargs = {"rgb": rgb,
                             "num_greedy": greedy,
                             "split": split,
                             "eject": eject}
                )


