def read(nexp, method, path, env, 
    columns = ['it','reward','time','nrollouts','nsimulations'],
    time_constrained=False, max_time_minutes=10):

    results = []
    for exp in range(nexp):
        # preparing results dict
        results.append({})
        for column in columns:
            results[-1][column] = []

        # reading the data
        with open(path+env+'_'+method+'_test_'+str(exp)+'.csv','r') as resultfile:
            count, running_time = 0, 0.0
            for line in resultfile:
                if count > 0:
                    fcolumns = line.split(';')
                    for i in range(len(columns)):
                        results[exp][columns[i]].append(float(fcolumns[i]))
                    
                    if time_constrained:
                        running_time += float(fcolumns[2])
                        if running_time > max_time_minutes*60:
                            break
                count += 1
    return results