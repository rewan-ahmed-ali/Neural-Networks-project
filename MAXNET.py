from random import uniform

def activation_fn(x):
    if x > 0:
        return x
    return 0

# Step 1 - Initialize weights and activations
activations = [0.3, 0.5, 0.7, 0.9]
m = len(activations)
k = 0.2

a_old = activations.copy()
a_new = []
count = 0

while True:  # Step 2
    temp = sum(a_old)
    
    for i in range(0, m):  # Step 3 - Update activations of each node
        value = a_old[i] - k * temp + k * a_old[i]
        a_new.append(activation_fn(value))
    
    a_old = a_new.copy()  # Step 4 - Save activations for use in each iteration
    
    if sum(a_new) == max(a_new):  # Step 5 - Test for stopping condition
        break
    
    a_new = []
    count += 1
    print('EPOCH {} - activations = {}'.format(count, a_old))

print('The final activations are {}'.format(a_new))
