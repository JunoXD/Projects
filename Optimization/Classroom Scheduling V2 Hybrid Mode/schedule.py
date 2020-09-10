import pandas as pd
import numpy as np
from datetime import datetime,date,timedelta,time
from gurobipy import Model, GRB
from itertools import combinations

def optimize(inputFile, outputFile):
    # Read input file
    sections=pd.read_excel(inputFile,sheet_name='Classes').set_index('SECTION')
    rooms=pd.read_excel(inputFile,sheet_name='Rooms',index_col=1)
    block=pd.read_excel(inputFile,sheet_name='Black Outs')

    # Step 1: Pair up half-semester sections
    print("Step 1: Pair up half-semester sections")
    # 𝐺 : set of first-half-semester sections
    # 𝐻: set of second-half-semester sections
    G=sections[sections['Sem Type']==1].index
    H=sections[sections['Sem Type']==2].index

    # phi: set of section pairs (𝑔,ℎ) that do not meet at the same time or with the same frequency every week
    # seatsDiffDF: seats offered difference between a first-half-semester section and a second-half-semester section
    # sameProfDF: whether a first-half-semester section and a second-half-semester section have the same instructor
    phi=[]
    seatsDiffDF=pd.DataFrame(index=G,columns=H) #u
    sameProfDF=pd.DataFrame(index=G,columns=H) #m
    for g in G:
        for h in H:
            if sections.loc[g,'FIRST DAYS']!=sections.loc[h,'FIRST DAYS'] or \
            sections.loc[g,'FIRST BEGIN TIME']!=sections.loc[h,'FIRST BEGIN TIME'] or\
            sections.loc[g,'FIRST END TIME']!=sections.loc[h,'FIRST END TIME']:
                phi.append((g,h))
            seatsDiffDF.loc[g,h]=max(sections.loc[g,'REG COUNT']-sections.loc[h,'REG COUNT'],
                                     sections.loc[h,'REG COUNT']-sections.loc[g,'REG COUNT'])
            if sections.loc[g,'FIRST INSTRUCTOR']==sections.loc[h,'FIRST INSTRUCTOR'] or\
            sections.loc[g,'FIRST INSTRUCTOR']==sections.loc[h,'SECOND INSTRUCTOR'] or\
            sections.loc[g,'SECOND INSTRUCTOR']==sections.loc[h,'SECOND INSTRUCTOR'] or\
            sections.loc[g,'SECOND INSTRUCTOR']==sections.loc[h,'FIRST INSTRUCTOR']:
                sameProfDF.loc[g,h]=1
    sameProfDF=sameProfDF.fillna(0)

    # Set up GUROBI model for paring up half-semester sections
    alpha=5 # extra weight on same professor pairs
    mod1=Model()
    l=mod1.addVars(G,H,vtype=GRB.BINARY)
    paired_secs=sum(l[g,h] for g in G for h in H)
    paired_secs_same_prof=sum(l[g,h]*sameProfDF.loc[g,h] for g in G for h in H)
    tot_seats_diff=sum(l[g,h]*seatsDiffDF.loc[g,h] for g in G for h in H)
    # Maximize the number of paired sections, with extra weights put on pair of sections taught by the same professor
    mod1.setObjective(paired_secs+alpha*paired_secs_same_prof,
                      sense=GRB.MAXIMIZE)
    # Only sections that meet at the same time and frequency every week are allowed to be paired
    for pair in phi:
        mod1.addConstr(l[pair[0],pair[1]]==0)
    # Each section can only be paired once
    for h in H:
        mod1.addConstr(sum(l[g,h] for g in G)<=1)
    for g in G:
        mod1.addConstr(sum(l[g,h] for h in H)<=1)
    mod1.optimize()

    # Print pairing resulds
    print('# paired sections',paired_secs.getValue())
    print('# paired sections w/ same instructor',paired_secs_same_prof.getValue())
    print('Total seats offered differences',tot_seats_diff.getValue())

    # Step 2: Assign classrooms
    print("-------------------------------------------------------------------")
    print("Step 2: Assign classrooms")
    sections['Core']=sections['Type General'].apply(lambda x: 1 if x!='Elective' else 0)
    secs=sections[['Course','FIRST DAYS','FIRST BEGIN TIME','FIRST END TIME',
           'FIRST INSTRUCTOR','SECOND INSTRUCTOR','REG COUNT','Core',]].copy()

    C=secs.Course.unique() # C: List of courses
    courseSecDict={} # I_c: set of sections that belong to course 𝑐∈𝐶
    courseSeatsDict={} # r_c: total seats offered by course 𝑐∈𝐶
    courseCoreDict={} # w_c: whether course 𝑐 is a core course
    for course in C:
        courseSecDict[course]=set(secs[secs['Course']==course].index)
        courseSeatsDict[course]=secs[secs['Course']==course]['REG COUNT'].sum()
        courseCoreDict[course]=secs[secs['Course']==course]['Core'].unique()[0]

    # P: set of professors
    P=set(list(secs['FIRST INSTRUCTOR'].unique())+list(secs['SECOND INSTRUCTOR'].unique()))
    P.remove(np.nan)
    profSecDict={} #I_p: set of sections taught by professor 𝑝∈𝑃
    for prof in P:
        profSecDict[prof]=list(set(list(secs[secs['FIRST INSTRUCTOR']==prof].index)+\
                              list(secs[secs['SECOND INSTRUCTOR']==prof].index)))

    # Combine paired half-semester sections into full semester sections
    # Unpaired hals-semester sections are treated as full semester sections
    for g in G:
        for h in H:
            if l[g,h].x==1:
                sec_name=str(g)+'/'+str(h)
                course1=secs.loc[g,'Course']
                course2=secs.loc[h,'Course']
                # Update course-section dictionary
                courseSecDict[course1].add(sec_name)
                courseSecDict[course2].add(sec_name)
                courseSecDict[course1].remove(g)
                courseSecDict[course2].remove(h)
                # Update professor-section dictionary
                for prof in [secs.loc[g,'FIRST INSTRUCTOR'],secs.loc[g,'SECOND INSTRUCTOR']]:
                    if isinstance(prof,str):
                        profSecDict[prof].append(sec_name)
                        profSecDict[prof].remove(g)
                for prof in [secs.loc[h,'FIRST INSTRUCTOR'],secs.loc[h,'SECOND INSTRUCTOR']]:
                    if isinstance(prof,str):
                        profSecDict[prof].append(sec_name)
                        profSecDict[prof].remove(h)
                secs.loc[g,'Combined']=1
                secs.loc[h,'Combined']=1
                secs.loc[sec_name,'FIRST DAYS']=secs.loc[g,'FIRST DAYS']
                secs.loc[sec_name,'FIRST BEGIN TIME']=secs.loc[g,'FIRST BEGIN TIME']
                secs.loc[sec_name,'FIRST END TIME']=secs.loc[g,'FIRST END TIME']

    final_secs=secs[secs['Combined'].isnull()][['FIRST DAYS','FIRST BEGIN TIME','FIRST END TIME']].copy()
    I=final_secs.index # I: list of sections
    K=rooms.index # K: list of classrooms
    roomLimit=30
    # Limit classroom capacity to 30 person
    cap=rooms['6ft'].fillna(10).apply(lambda x: roomLimit if x>roomLimit else x).astype(int) #m_k

    # This function translates time into minutes since 12:00am for easy comparison
    def minutes(text):
        return int(text[:2])*60+int(text[3:5])

    # 𝑇 : set of section pairs (𝑖1,𝑖2) that are held at the same or overlapping time slots.
    # 𝐹 : set of section pairs (𝑖1,𝑖2) such that section 𝑖1 and 𝑖2 are taught by the same professor and the interval between two sections is less than 20 minutes.
    T=[]
    F=[]
    secPairList=list(combinations(I,2))
    for (i,ii) in secPairList:
        start1=final_secs.loc[i,'FIRST BEGIN TIME']
        start2=final_secs.loc[ii,'FIRST BEGIN TIME']
        end1=final_secs.loc[i,'FIRST END TIME']
        end2=final_secs.loc[ii,'FIRST END TIME']
        day1=final_secs.loc[i,'FIRST DAYS']
        day2=final_secs.loc[ii,'FIRST DAYS']
        if day1==day2 or day1 in day2 or day2 in day1:
            if (minutes(start1)>=minutes(start2) and minutes(start1)<minutes(end2)) or\
            (minutes(start2)>=minutes(start1) and minutes(start2)<minutes(end1)):
                T.append((i,ii))
            elif (minutes(start1)-minutes(end2)>=0 and minutes(start1)-minutes(end2)<=20) or \
            (minutes(start2)-minutes(end1)>=0 and minutes(start2)-minutes(end1)<=20):
                profList1=set()
                profList2=set()
                for prof, sec in profSecDict.items():
                    if i in sec:
                        profList1.add(prof)
                    if ii in sec:
                        profList2.add(prof)
                if len(profList1&profList2)>0:
                    F.append((i,ii))

    # 𝑁 : set of section-classroom pairs (𝑖,𝑘) such that the time when section 𝑖 is held is blocked out for classroom 𝑘
    N=[]
    for idx in block.index:
        k=block.loc[idx,'Room']
        block_start=minutes(block.loc[idx,'Start Time'])
        block_end=minutes(block.loc[idx,'End Time'])
        block_days=block.loc[idx,'Days']
        for day in block_days:
            pop_slots=final_secs[final_secs['FIRST DAYS'].str.contains(day)].copy()
            pop_slots['Block Start']=block_start
            pop_slots['Block End']=block_end
            pop_slots['Start']=pop_slots['FIRST BEGIN TIME'].apply(lambda x: minutes(x))
            pop_slots['End']=pop_slots['FIRST END TIME'].apply(lambda x: minutes(x))
            pop_slots['Blocked']=pop_slots[['Block Start','Block End','Start','End']].apply(lambda x:
                                1 if (x[2]>=x[0] and x[2]<x[1]) or (x[0]>=x[2] and x[0]<x[3]) else 0, axis=1)
            for i in pop_slots[pop_slots['Blocked']==1].index:
                N.append((i,k))

    B=rooms.Building.unique().tolist() # 𝐵 : set of buildings
    bldgRoomDict={} # K_b: set of classrooms in regions  𝑏∈𝐵
    for b in B:
        bldgRoomDict[b]=rooms[rooms['Building']==b].index.tolist()

    sigma1=2 # base weight for Hybrid50 mode
    sigma2=5 # additional weight for Hybrid50 mode if it's a core course
    sigma3=1 # weight for Hybrid33 mode
    e=10000 # arbitrary large number (larger than the sum of seats offered for any given course)

    # Set up GUROBI model to assign classrooms
    start_time = pd.datetime.now()
    mod2=Model()
    x=mod2.addVars(I,K,vtype=GRB.BINARY)
    y=mod2.addVars(C,vtype=GRB.BINARY)
    z=mod2.addVars(C,vtype=GRB.BINARY)
    q=mod2.addVars(I,B,vtype=GRB.BINARY)
    hybrid50_seats=sum(y[c]*courseSeatsDict[c] for c in C)
    hybrid50_core_seats=sum(y[c]*courseCoreDict[c]*courseSeatsDict[c] for c in C)
    hybrid33_seats=sum((z[c]-y[c])*courseSeatsDict[c] for c in C)
    # Maximize the seats in Hybrid50 and Hybrid33 mode classes (weighted differently based on 50/33, core/elective)
    mod2.setObjective(sigma1*hybrid50_seats+sigma2*hybrid50_core_seats+sigma3*hybrid33_seats,
                      sense=GRB.MAXIMIZE)
    print('Model initiated and variables added:',pd.datetime.now()-start_time,flush=True)

    #1 Classroom capacity
    start_time = pd.datetime.now()
    for c in C:
        mod2.addConstr(2*sum(x[i,k]*cap[k] for i in courseSecDict[c] for k in K)\
                       >=courseSeatsDict[c]*y[c])
        mod2.addConstr(3*sum(x[i,k]*cap[k] for i in courseSecDict[c] for k in K)<=e*z[c])
        mod2.addConstr(3*sum(x[i,k]*cap[k] for i in courseSecDict[c] for k in K)\
                       >=courseSeatsDict[c]*z[c])
    print('Constraint #1 added:',pd.datetime.now()-start_time,flush=True)

    #2 Core hybrid50
    start_time = pd.datetime.now()
    for c in C:
        if courseCoreDict[c]==1:
            mod2.addConstr(z[c]==y[c])
    print('Constraint #2 added:',pd.datetime.now()-start_time,flush=True)

    #3 Same section conflict
    start_time = pd.datetime.now()
    for i in I:
        mod2.addConstr(sum(x[i,k] for k in K)<=1)
    print('Constraint #3 added:',pd.datetime.now()-start_time,flush=True)

    #4 Same classroom conclict
    start_time = pd.datetime.now()
    for k in K:
        for (i1,i2) in T:
            mod2.addConstr(x[i1,k]+x[i2,k]<=1)
    print('Constraint #4 added:',pd.datetime.now()-start_time,flush=True)

    #5 Blackout time
    start_time = pd.datetime.now()
    for (i,k) in N:
        mod2.addConstr(x[i,k]==0)
    print('Constraint #5 added:',pd.datetime.now()-start_time,flush=True)

    # 6 Proximity
    start_time = pd.datetime.now()
    for i in I:
        for b in B:
            mod2.addConstr(q[i,b]==sum(x[i,k] for k in bldgRoomDict[b]))
        for (i1,i2) in F:
            BB=B.copy()
            BB.remove(b)
            mod2.addConstr(q[i1,b]+sum(q[i2,bb] for bb in BB)<=1)
    print('Constraint #6 added:',pd.datetime.now()-start_time,flush=True)

    start_time = pd.datetime.now()
    mod2.optimize()
    print('Model optimized:',pd.datetime.now()-start_time,flush=True)

    # Read assigned classrooms
    for i in I:
        for k in K:
            if x[i,k].x==1:
                if isinstance(i,str):
                    sec_pair=i.split('/')
                    for sec in sec_pair:
                        sections.loc[int(sec),'Room']=k
                else:
                    sections.loc[i,'Room']=k

    # Determine hybrid mode
    hybrid=pd.Series(index=C,name='Type')
    for c in C:
        if y[c].x==1:
            hybrid[c]='hybrid50'
        elif z[c].x==1:
            hybrid[c]='hybrid33'
        else:
            hybrid[c]='online'

    mg=sections.merge(cap,how='left',left_on='Room',right_index=True)
    mg=mg.merge(hybrid,left_on='Course',right_index=True,how='left')
    mg['Core']=mg['Core'].map({0:'Elective',1:'Core'})

    # Create seat count summary
    summarySeats=mg.groupby(['Core','Type']).sum()['REG COUNT'].reset_index(name='Seats Offered')
    summarySeats=pd.pivot_table(summarySeats,index=['Core'],columns='Type',values='Seats Offered').fillna(0)
    summarySeats=summarySeats.astype(int)
    summarySeats.loc['Total']=summarySeats.sum()
    summarySeats['Total']=summarySeats.sum(axis=1)
    summarySeats['hybrid33 %']=(summarySeats['hybrid33']/summarySeats['Total']).apply(lambda x: f'{x*100:.1f}%')
    summarySeats['hybrid50 %']=(summarySeats['hybrid50']/summarySeats['Total']).apply(lambda x: f'{x*100:.1f}%')
    summarySeats['online %']=(summarySeats['online']/summarySeats['Total']).apply(lambda x: f'{x*100:.1f}%')

    # Create course count summary
    summaryCourses=mg.groupby(['Core','Type']).nunique()['Course'].reset_index(name='# Courses')
    summaryCourses=pd.pivot_table(summaryCourses,index=['Core'],columns='Type',values='# Courses').fillna(0)
    summaryCourses=summaryCourses.astype(int)
    summaryCourses.loc['Total']=summaryCourses.sum()
    summaryCourses['Total']=summaryCourses.sum(axis=1)
    summaryCourses['hybrid33 %']=(summaryCourses['hybrid33']/summaryCourses['Total']).apply(lambda x: f'{x*100:.1f}%')
    summaryCourses['hybrid50 %']=(summaryCourses['hybrid50']/summaryCourses['Total']).apply(lambda x: f'{x*100:.1f}%')
    summaryCourses['online %']=(summaryCourses['online']/summaryCourses['Total']).apply(lambda x: f'{x*100:.1f}%')

    print("-------------------------------------------------------------------")
    print("Optimization Complete")
    # Print summaries to screen
    print(summarySeats)
    print(summaryCourses)

    # Write results to output file
    writer=pd.ExcelWriter(outputFile)
    mg.to_excel(writer,sheet_name='Schedules',index=True)
    summarySeats.to_excel(writer,sheet_name='Seats Summary')
    summaryCourses.to_excel(writer,sheet_name='Courses Summary')
    writer.save()

if __name__=='__main__':
    import sys, os
    if len(sys.argv)!=3:
        print('Correct syntax: python schedule.py inputFile outputFile')
    else:
        inputFile=sys.argv[1]
        outputFile=sys.argv[2]
        if os.path.exists(inputFile):
            optimize(inputFile,outputFile)
            print(f'Successfully optimized. Results in "{outputFile}"')
        else:
            print('Input file not Found!')
