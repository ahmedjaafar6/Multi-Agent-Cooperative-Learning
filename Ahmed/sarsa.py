import numpy as np
import matplotlib.pyplot as plt
import time


height = 6
width = 5

a_map = {'>': 0, '<': 1, '^': 2, 'v': 3}
q_hat1= np.zeros(shape=(height,width,4)) #q(s,a) intialized to 0
q_hat2= np.zeros(shape=(height,width,4)) #q(s,a) intialized to 0
policy_hat1 = np.full(shape=(height,width,4), fill_value=0.25) #has probablities for each state-action pair
policy_hat1[1,2], policy_hat1[3,2], policy_hat1[4,2], policy_hat1[2,3] = 0,0,0,0
policy_hat2 = np.full(shape=(height,width,4), fill_value=0.25) #has probablities for each state-action pair
policy_hat2[1,2], policy_hat2[3,2], policy_hat2[4,2], policy_hat2[2,3]  = 0,0,0,0
value_hat1 = np.zeros(shape=(height,width))
value_hat2 = np.zeros(shape=(height,width))


def show_policy(q_hat):
    a_map2 = {0: '🠮', 1: '🠬 ', 2: '🠭', 3: '🠯'}
    policy = np.empty(shape=(height,width), dtype='<U1')
    a_star_mat = np.argmax(q_hat, axis=2) #2d matrix of 3rd dim highest index
    
    for y in range(height):
        for x in range(width):
            if (x == 2 and (y == 1 or y == 3 or y == 4)) or (y == 2 and x == 3):
                policy[y,x] = '#'
                continue
            policy[y,x] = a_map2[a_star_mat[y,x]]
    
    policy[5,4] = "G"

    return policy


def run_episode(epsilon1, epsilon2, alpha, gamma1, gamma2):
    global policy_hat1
    global policy_hat2
    global q_hat1
    global q_hat2
    
    global doorAgent1
    global doorAgent2
    doorAgent1, doorAgent2 = False, False
    global agent1 
    agent1 = [2,0] #position agent 1
    global agent2 
    agent2 = [5,0] #position agent 2
    global reward
    reward = 0

    def step(policy_hat, agent):
        global reward
        global doorAgent1
        global doorAgent2
        global agent1
        global agent2

        if agent == 1:
            curr_y , curr_x = agent1
            curr_agent = 1
        else:
            curr_y, curr_x = agent2
            curr_agent = 2
        

        a0_prob = policy_hat[curr_y, curr_x, 0] #right
        a1_prob = policy_hat[curr_y, curr_x, 1] #left
        a2_prob = policy_hat[curr_y, curr_x, 2] #up
        a3_prob = policy_hat[curr_y, curr_x, 3] #down
        
        action = np.random.choice(['>','<','^','v'], p=[a0_prob, a1_prob, a2_prob, a3_prob])
        next_y, next_x = 0, 0

        if action == '>':
            if curr_x == width-1 or (curr_x==1 and (curr_y==1 or curr_y==3 or curr_y==4)) or (curr_x==2 and curr_y==2) or (doorAgent1==False and doorAgent2==False and curr_y == 5 and curr_x == 1):
                next_y = curr_y
                next_x = curr_x
            else:
                next_y = curr_y
                next_x = curr_x + 1
        elif action == '<':
            if curr_x == 0 or (curr_x == 3 and (curr_y==1 or curr_y==3 or curr_y==4)) or (curr_x == width-1 and curr_y == 2) or (doorAgent1==False and doorAgent2==False and curr_y == 5 and curr_x == 3): 
                next_y = curr_y
                next_x = curr_x
            else:
                next_y = curr_y
                next_x = curr_x - 1
        elif action == '^':
            if curr_y == 0 or (curr_y==2 and curr_x==2) or (curr_y==3 and curr_x==3) or (curr_y==5 and curr_x==2):
                next_y = curr_y
                next_x = curr_x
            else:
                next_y = curr_y-1
                next_x = curr_x
        elif action == 'v':
            if curr_y == height-1 or (curr_y==2 and curr_x==2) or (curr_y==1 and curr_x==3) or (curr_y==0 and curr_x==2):
                next_y = curr_y
                next_x = curr_x
            else:
                next_y = curr_y+1
                next_x = curr_x
        
        #check if other agent in the next state, if it is, treat it as a wall
        if curr_agent == 1:
            if next_y == agent2[0] and next_x == agent2[1]:
                next_y = curr_y
                next_x = curr_x
        else:
            if next_y == agent1[0] and next_x == agent1[1]:
                next_y = curr_y
                next_x = curr_x

            
        #updating agent position
        if agent == 1:
            agent1 = [next_y, next_x]
        else:
            agent2 = [next_y, next_x]

        #opening door
        if agent1[0] == 2 and agent1[1] == 2 and curr_agent == 1: #if agent1 on button
            doorAgent1 = True #open door
        elif agent2[0] == 2 and agent2[1] == 2 and curr_agent == 2: #if agent2 on button
            doorAgent2 = True #open door
        
        #closing door
        if agent1[0] != 2 and agent1[1] != 2 and curr_agent == 1:
            doorAgent1 = False
        elif agent2[0] != 2 and agent2[1] != 2 and curr_agent == 2:
            doorAgent2 = False

        if next_y > 5 or next_y < 0 or next_x > 4 or next_x < 0:
            print(action)
            print(curr_y, curr_x)
            print(f"WARNING: NEXT STATE OUT OF BOUNDS! next_y = {next_y}, next_x = {next_x}")

        #rewards 
        if next_y == 5 and next_x == 4: #goal state
            reward += 10
        else:
            reward += 0

        a0_next_prob = policy_hat[next_y, next_x, 0] #right
        a1_next_prob = policy_hat[next_y, next_x, 1] #left
        a2_next_prob = policy_hat[next_y, next_x, 2] #up
        a3_next_prob = policy_hat[next_y, next_x, 3] #down

        next_action = np.random.choice(['>','<','^','v'], p=[a0_next_prob, a1_next_prob, a2_next_prob, a3_next_prob])
    

        #if it reaches the goal state
        if next_y == 5 and next_x == 4:
            return 'end', action, next_action, policy_hat, curr_y, curr_x
        
        return 'continue', action, next_action, policy_hat, curr_y, curr_x

        ##########################################################



    count = 0
    while True:
        step1, action1, next_action1, policy_hat1_return, curr_y1, curr_x1 = step(policy_hat1, 1) #agent 1 takes a step
        step2, action2, next_action2, policy_hat2_return, curr_y2, curr_x2 = step(policy_hat2, 2) #agent 2 takes a step
        policy_hat1, policy_hat2 = policy_hat1_return, policy_hat2_return
        count += 1

        #agent 1 q updates
        curr_q = q_hat1[curr_y1, curr_x1, a_map[action1]]
        next_q = q_hat1[agent1[0], agent1[1], a_map[next_action1]]

        q_hat1[curr_y1, curr_x1, a_map[action1]] = curr_q + (alpha * ((reward+(gamma1*next_q)) - curr_q))
        
        max_a_indx = np.argmax(q_hat1[curr_y1, curr_x1]) #max action index

        #updating policy_hat1 (policy improvement)
        for i in range(4):
            max_value = q_hat1[curr_y1, curr_x1, max_a_indx]
            A_star_ct = np.count_nonzero(q_hat1[curr_y1, curr_x1, :] == max_value) #count of max values
            if q_hat1[curr_y1, curr_x1, i] == max_value:
                policy_hat1[curr_y1,curr_x1,i] = ((1-epsilon1)/A_star_ct)+(epsilon1/4)
            else:
                policy_hat1[curr_y1,curr_x1,i] = epsilon1/4

        #agent 2 q updates
        curr_q = q_hat2[curr_y2, curr_x2, a_map[action2]]
        next_q = q_hat2[agent2[0], agent2[1] , a_map[next_action2]]

        q_hat2[curr_y2, curr_x2, a_map[action2]] = curr_q + (alpha * ((reward+(gamma2*next_q)) - curr_q))
        
        max_a_indx = np.argmax(q_hat2[curr_y2, curr_x2]) #max action index

        #updating policy_hat2 (policy improvement)
        for i in range(4):
            max_value = q_hat2[curr_y2, curr_x2, max_a_indx]
            A_star_ct = np.count_nonzero(q_hat2[curr_y2, curr_x2, :] == max_value) #count of max values
            if q_hat2[curr_y2, curr_x2, i] == max_value:
                policy_hat2[curr_y2, curr_x2,i] = ((1-epsilon2)/A_star_ct)+(epsilon2/4)
            else:
                policy_hat2[curr_y2, curr_x2,i] = epsilon2/4
        
        if step1 == 'end' or step2 == 'end':
            break

    return count






start = time.time()
avg_steps = np.zeros(shape=(20,20000))
for i in range(20):
    print(f'Iteration {i+1}')

    epsilon1 = 0.4 #0.2
    epsilon2 = 0.3
    alpha = 0.2 #0.2
    gamma1 = 0.8
    gamma2 = 0.7

    step_count = 0
    steps_lst = np.zeros(shape=(20000))
    for j in range(20000): #episodes
        if j % 1000 == 0:
            print(j, end="\r")
        
        if j % 500 == 0:
            epsilon1 *= 0.98

        count = run_episode(epsilon1, epsilon2, alpha, gamma1, gamma2)

        step_count += count
        steps_lst[j] = step_count
        
        # estimating the value function using q and pi
        for y in range(height):
            for x in range(width):
                value_hat1[y,x] = np.sum(np.multiply(q_hat1[y,x,:], policy_hat1[y,x,:]))
                value_hat2[y,x] = np.sum(np.multiply(q_hat2[y,x,:], policy_hat2[y,x,:]))
    
    avg_steps[i] = steps_lst

avg_steps = np.mean(avg_steps, axis=0)


plt.xlabel("Steps")
plt.ylabel("Episodes")
plt.plot(avg_steps, np.arange(20000)+1)
plt.show()



end = time.time()
print(f'\nTime: {(end-start)/60}')