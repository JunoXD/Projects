import pandas as pd
from datetime import datetime, time, date, timedelta
from gurobipy import Model, GRB
import itertools as it
import math

def optimize(inputFile, outputFile):
    """Schedules the sections based on the classrooms, standard slots and professor preference provided. Professor preference is optional."""
    tot_start_time=pd.datetime.now()
    # read all input files
    print('-->>Start reading input file...',flush=True)
    sections=pd.read_excel(inputFile,sheet_name='Sections').set_index('section')
    slots=pd.read_excel(inputFile,sheet_name='StandardSlots',index_col=0)
    cap=pd.read_excel(inputFile,sheet_name='ClassroomCapacity',index_col=0)['Size']
    groups=pd.read_excel(inputFile,sheet_name='TogetherCourses')
    avai={}
    # create a template preference/availability table with all time slots being available
    # this will be used if the preference/availability of a professor is not provided
    hourList=[str((datetime.combine(date.today(),time(8,0))+timedelta(minutes=30*i)).time()) for i in range(27)]
    dayList=[f'{i}{j}' for i in ['M','T','W','H','F'] for j in [1,2]]
    fakeAvai=pd.DataFrame(1,index=hourList,columns=dayList)

    try:
        # check if the ProfessorPreference sheet is provided at all
        avaiInput=pd.read_excel(inputFile,sheet_name='ProfessorPreference',index_col=0)
        prof_with_avai=avaiInput.index.unique()
        for prof in sections.instructor.unique():
            # check if any section has an instructor whose preference is not provided
            # if so, use the all-available template
            if prof in prof_with_avai:
                avai[prof]=avaiInput.loc[prof].set_index('time')
            else:
                avai[prof]=fakeAvai
    except:
        # if the ProfessorPreference sheet does not exist
        # use the all-available template for all professors
        for prof in sections.instructor.unique():
            avai[prof]=fakeAvai
    print('Input file processed',flush=True)

    # calculate the number of sections need to be scheduled
    numSec=len(sections)
    # obtain the list of courses that are involed for future reference
    # some course will later be combined when pairing up half semester sections
    original_courses=sections.course.unique()

    def decide_slot_length(length, lengthList):
        """Determine the smallest possible slot (represented by multiple of 30 minutes) that the section can fit in"""
        lengthList.sort()
        for l in lengthList:
            if length<=l*30:
                return l
        else:
            return 0

    # Determine the smallest possible slot that the section can fit in
    lengthList=slots['#half_hrs'].unique()
    ss=sections.copy()
    ss['#half_hrs']=ss['length'].apply(lambda x: decide_slot_length(x, lengthList))
    ss=ss[ss['#half_hrs']!=0][['course','meeting_frequency','session','instructor','seats_offered','#half_hrs']]

    # Calculate the total length of sections that need to be scheduled
    totSecLength=(ss['#half_hrs']*ss['meeting_frequency']*\
             ss['session'].apply(lambda x: 1 if x in [1,33] else 0.5)*30).sum()/60
    # Calculate the total time available all the classrooms combined
    slots2=slots.copy()
    slots2['day']=slots2.index.str[:-11]
    slots2['hour']=slots2.index.str[-10:-8].astype(int)
    slots2['minute']=slots2.index.str[-7:-5].astype(int)
    slots2['time']=slots2[['hour','minute']].apply(lambda x: time(x[0],x[1]),axis=1)
    slots2=slots2.set_index('day')
    lengths={}
    for day in slots2.index.unique():
        df=slots2.loc[day]
        start=df['time'].min()
        end_start=df['time'].max()
        end_length=df[df.time==end_start]['#half_hrs'].max()
        length=(datetime.combine(date.min,end_start)+\
               timedelta(minutes=end_length*30)-datetime.combine(date.min,start)).total_seconds()/60
        if len(day)==1:
            lengths[day]=length
        else:
            for d in day:
                if d not in lengths.keys() or lengths[d]<length:
                    lengths[d]=length
    tot_length=0
    for l in lengths.values():
        tot_length+=l
    totClassroomLength=tot_length/60*len(cap)
    print('Total length of sections need to be scheduled:',totSecLength,'hrs',flush=True)
    print('Section lengths are rounded to the length of the smallest standard slot it fits in.')
    print('Total length of time available all classrooms combined:',totClassroomLength,'hrs',flush=True)
    print(f'Demand/resources: {totSecLength/totClassroomLength*100:.1f}%',flush=True)
    print(flush=True)
    print('------------------------------------------------------------------',flush=True)
    print('------------------------------------------------------------------',flush=True)
    print(flush=True)

    # # Part I: Pair Up Half-Semester Sections
    print('-->>Start paring up half-semester sections...',flush=True)
    firstHalf=[411, 431]
    secondHalf=[415, 442]
    full=[1, 33]
    II=ss[ss['session'].isin(firstHalf)].index # first-half sections
    JJ=ss[ss['session'].isin(secondHalf)].index # second-half sections

    # Calculate the combined preference score for the pair of professors towards a time slot
    sss=pd.DataFrame(index=II,columns=JJ)
    prof_avai2=pd.DataFrame(columns=slots.index)
    for i in II:
        for j in JJ:
            # only pair up sections with the same meeting frequency and length
            if ss.loc[i,'meeting_frequency']!=ss.loc[j,'meeting_frequency'] or \
            ss.loc[i,'#half_hrs']!=ss.loc[j,'#half_hrs']:
                sss.loc[i,j]=0
            else:
                prof1=ss.loc[i,'instructor']
                prof2=ss.loc[j,'instructor']
                profs=f'{prof1}_{prof2}'
                # combine the first half semester preference table of the prefessor
                # who's teaching the first half semester
                # with the second half semester preference table of the prefessor
                # who's teaching the second half semester
                profDF1=avai[prof1].copy()
                col1=[col for col in profDF1.columns if '1' in col]
                profDF1=profDF1[col1]
                profDF1.columns=[col[0] for col in col1]
                profDF2=avai[prof2].copy()
                col2=[col for col in profDF2.columns if '2' in col]
                profDF2=profDF2[col2]
                profDF2.columns=[col[0] for col in col2]
                # calculate average peference of the two professors
                df=(profDF1+profDF2)/2
                # if any of the professor is not available at a time
                # set the slot preference to 0
                df[(profDF1==0) | (profDF2==0)]=0

                # calculate the preference score for each standard time slots
                for slot in slots[slots['meeting_frequency']==1].index:
                    day=slot[0]
                    startH=int(slot[2:4])
                    startM=int(slot[5:7])
                    start=time(startH,startM)
                    numHalf=int(slot[11:12])
                    # do not consider slots that have different length
                    # from the sections under consideration
                    if numHalf!=ss.loc[i,'#half_hrs']:
                        unavai=True
                    else:
                        unavai=False
                        tot=0
                        count=0
                        for n in range(numHalf):
                            curTime=(datetime.combine(date.today(),start)+timedelta(minutes=30*n)).time()
                            # if any partial 30 minutes of the standard slot doesn't work for the professor
                            # set to unavailable
                            if df.loc[str(curTime),day]==0:
                                unavai=True
                                break
                            else:
                                tot+=df.loc[str(curTime),day]
                                count+=1
                    if unavai:
                        prof_avai2.loc[profs,slot]=0
                    else:
                        prof_avai2.loc[profs,slot]=tot/count
                # for slots that meet twice a week, make it available only when
                # both days and time work for the professors
                if ss.loc[i,'meeting_frequency']==2:
                    for slot in slots[slots['meeting_frequency']==2].index:
                        days=slot[:2]
                        parts=slot[2:]
                        if prof_avai2.loc[profs,days[0]+parts]>0 and prof_avai2.loc[profs,days[1]+parts]>0:
                            prof_avai2.loc[profs,slot]=(prof_avai2.loc[profs,days[0]+parts]+ \
                            prof_avai2.loc[profs,days[1]+parts])/2
                        else:
                            prof_avai2.loc[profs,slot]=0
                    prof_avai2.loc[profs,slots[slots['meeting_frequency']==1].index]=0
                else:
                    prof_avai2.loc[profs,slots[slots['meeting_frequency']==2].index]=0
                sss.loc[i,j]=(prof_avai2.loc[profs]>0).sum()

    #set up Gurobi optimization
    mod0=Model()
    xx=mod0.addVars(II,JJ,vtype=GRB.BINARY)
    mod0.setObjective(sum(sss.loc[i,j]*xx[i,j] for i in II for j in JJ),
                      sense=GRB.MAXIMIZE)
    # ineligible slots
    for i in II:
        for j in JJ:
            if sss.loc[i,j]==0:
                mod0.addConstr(xx[i,j]==0)
        # each first half semester section can only be paired once
        mod0.addConstr(sum(xx[i,j] for j in JJ)<=1)
    for j in JJ:
        # each second half semester section can only be paired once
        mod0.addConstr(sum(xx[i,j] for i in II)<=1)

    # set a time limit to force gurobi to end
    # mod0.Params.TimeLimit = 5*60
    mod0.optimize()

    # store the paired sections in a list
    paired=[]
    for i in II:
        for j in JJ:
            if xx[i,j].x>0:
                paired.append([i,j])
    print(flush=True)
    print('------------------------------------------------------------------',flush=True)
    print('------------------------------------------------------------------',flush=True)
    print(flush=True)
    print(f'Half-semester sections paired: {len(paired)*2} (out of {len(II)+len(JJ)})',flush=True)
    print('Time elapsed:', pd.datetime.now()-tot_start_time,flush=True)


    # # Part II
    print(flush=True)
    print('-->>Start preparing input for scheduling...',flush=True)
    start_time=pd.datetime.now()
    pairedCourses={}
    pairedProfs={}
    # alter the sections table so that each pair of paired sections
    # are replaced with one combined section
    # and store the paired courses and professors for future reference
    for pair in paired:
        df=ss.loc[pair]
        ss.loc['_'.join(pair)]=['_'.join(df['course']),
                                df['meeting_frequency'][0],
                                1,
                                '_'.join(df['instructor']),
                                df['seats_offered'].max(),
                                df['#half_hrs'][0]]
        pairedCourses['_'.join(df['course'])]=list(df['course'])
        pairedProfs['_'.join(df['instructor'])]=list(df['instructor'])
        ss=ss.loc[~ss.index.isin(pair)].copy()

    I=ss.index
    J=slots.index
    K=cap.index

    # S:set of section groups that belong to the same courses
    # if the two sections in a combined section belong to two courses
    # this combined section belongs to both courses and will appear up
    # in both courses' corresponding set of sections
    S={}
    for c in ss.course.unique():
        S[c]=set(ss[ss.course==c].index)
    for course in original_courses:
        for pair in pairedCourses.keys():
            if course in pair:
                if course in S.keys():
                    S[course].update(S[pair])
                else:
                    S[course]=S[pair]
    for pair in pairedCourses.keys():
        S.pop(pair)

    # Extract the available time slots and days from the slots input
    lengthSlots={}
    for n in slots['#half_hrs'].unique():
        tmpList=list(slots[slots['#half_hrs']==n].index)
        slotList=set()
        for t in tmpList:
            hour=int(t[-10:-8])
            minute=int(t[-7:-5])
            slotList.add(time(hour,minute))
        lengthSlots[n]=slotList
    days=pd.Series(slots.index).str[:-11].unique()

    # Find all overlapping time slots within the same day
    lengths=list(lengthSlots.keys())
    conflicts=[]
    for i in range(len(lengths)-1):
        for j in range(i+1,len(lengths)):
            l1=int(lengths[i])
            l2=int(lengths[j])
            slotList1=lengthSlots[l1]
            slotList2=lengthSlots[l2]
            for t1 in slotList1:
                t1_end=(datetime.combine(date.today(),t1)+timedelta(minutes=30*l1)).time()
                for t2 in slotList2:
                    t2_end=(datetime.combine(date.today(),t2)+timedelta(minutes=30*l2)).time()
                    if t1==t2 or (t2>t1 and t2<t1_end) or (t1>t2 and t1<t2_end):
                        conflicts.append([str(t1)+f'-{l1}',str(t2)+f'-{l2}'])

    # Find all overlapping time slots accross multiple days (one-day vs two-day slots)
    D=[]
    conflicted=set()
    pairs={'M':'MW','T':'TH','W':'MW','H':'TH'}
    for conf in conflicts:
        t1=conf[0]
        t2=conf[1]
        for day in days:
            if day in pairs.keys():
                dayList=[day,pairs[day]]
            elif len(day)==2:
                dayList=[day,day[0],day[1]]
            else:
                dayList=[day]

            ext_t1=day+'-'+t1
            if ext_t1 in J:
                for d in dayList:
                    ext_t2=d+'-'+t2
                    if ext_t2 in J:
                        D.append([ext_t1,ext_t2])
                        conflicted.add(ext_t1)
                        conflicted.add(ext_t2)
    for slot in slots[slots['meeting_frequency']==2].index:
        days=slot[:2]
        hours=slot[3:]
        for day in days:
            slot2=day+'-'+hours
            if slot2 in J:
                D.append([slot,slot2])
                conflicted.add(slot)
                conflicted.add(slot2)

    # list of slots that do not overlap with any other slots
    # this list is likely to be empty because usually most slots
    # overlap with some other slots in one way or another
    # (2hr vs 3hr vs 1.5hr, one-day vs two-day)
    U=[]
    for j in J:
        if j not in conflicted:
            U.append(j)

    # calculate professor preference towards standard time slots
    prof_avai=pd.DataFrame(index=ss.instructor.unique(),columns=slots.index)
    for instructor in ss.instructor.unique():
        if instructor in avai.keys():
            # meaning that the instructor is in fact ONE instructor
            # rather than TWO instructors who are teaching a combined section
            # the latter would not appear in the original preference input
            # combine half semester preferences into full semester
            dff=avai[instructor]
            col1=[col for col in dff.columns if '1' in col]
            col2=[col for col in dff.columns if '2' in col]
            cols=[col[0] for col in col1]
            df1=dff[col1]
            df1.columns=cols
            df2=dff[col2]
            df2.columns=cols
            df=((df1+df2)/2)[(df1>0)&(df2>0)].fillna(0)

            for slot in slots[slots['meeting_frequency']==1].index:
                day=slot[0]
                startH=int(slot[2:4])
                startM=int(slot[5:7])
                start=time(startH,startM)
                numHalf=int(slot[11:12])
                unavai=False
                tot=0
                count=0
                for n in range(numHalf):
                    curTime=(datetime.combine(date.today(),start)+timedelta(minutes=30*n)).time()
                    if df.loc[str(curTime),day]==0:
                        unavai=True
                        break
                    else:
                        tot+=df.loc[str(curTime),day]
                        count+=1
                if unavai:
                    prof_avai.loc[instructor,slot]=0
                else:
                    prof_avai.loc[instructor,slot]=tot/count
            for slot in slots[slots['meeting_frequency']==2].index:
                days=slot[:2]
                parts=slot[2:]
                if prof_avai.loc[instructor,days[0]+parts]>0 and prof_avai.loc[instructor,days[1]+parts]>0:
                    prof_avai.loc[instructor,slot]=(prof_avai.loc[instructor,days[0]+parts]+\
                    prof_avai.loc[instructor,days[1]+parts])/2
                else:
                    prof_avai.loc[instructor,slot]=0
        else:
            prof_avai.loc[instructor]=prof_avai2.loc[instructor]

    #a_ij: the preference score of the professor of section i towards time j
    a=ss[['instructor','session']].copy()
    a=pd.merge(a,prof_avai,left_on='instructor',right_index=True)
    # recalculate the preference score of the professors teaching half semester Sections
    # the score of the other half of the semester doesn't matter!!
    for i in a[-a.session.isin(full)].index:
        instructor=a.loc[i,'instructor']
        session=a.loc[i,'session']
        # only keep the corresponding half of the semester
        df=avai[instructor]
        col1=[col for col in df.columns if '1' in col]
        col2=[col for col in df.columns if '2' in col]
        cols=[col[0] for col in cols]
        if session in firstHalf:
            df=df[col1]
        else:
            df=df[col2]
        df.columns=cols

        for slot in slots[slots['meeting_frequency']==1].index:
            day=slot[0]
            startH=int(slot[2:4])
            startM=int(slot[5:7])
            start=time(startH,startM)
            numHalf=int(slot[11:12])
            unavai=False
            tot=0
            count=0
            for n in range(numHalf):
                curTime=(datetime.combine(date.today(),start)+timedelta(minutes=30*n)).time()
                if df.loc[str(curTime),day]==0:
                    unavai=True
                    break
                else:
                    tot+=df.loc[str(curTime),day]
                    count+=1
            if unavai:
                a.loc[i,slot]=0
            else:
                a.loc[i,slot]=tot/count
        for slot in slots[slots['meeting_frequency']==2].index:
            days=slot[:2]
            parts=slot[2:]
            if a.loc[i,days[0]+parts]>0 and a.loc[i,days[1]+parts]>0:
                a.loc[i,slot]=(a.loc[i,days[0]+parts]+a.loc[i,days[1]+parts])/2
            else:
                a.loc[i,slot]=0
    a=a.drop(columns=['instructor','session'])

    # A : set of section-time pairs (i,j) such that
    # either time slot j does not work for the professor of section i,
    # or section i and time slot j have different length or meeting frequency
    A=[]
    for i in I:
        for j in J:
            if ss.loc[i,'meeting_frequency']!=slots.loc[j,'meeting_frequency'] or\
            ss.loc[i,'#half_hrs']!=slots.loc[j,'#half_hrs'] or a.loc[i,j]==0:
                A.append([i,j])

    # H: set of section-classroom pairs (i,k) such that section i cannot fit in classroom k
    H=[]
    for i in I:
        for k in K:
            if ss.loc[i,'seats_offered']>cap.loc[k]:
                H.append([i,k])

    # P : set of section groups that are taught by the same professor
    # if the two sections in a combined section have different professors
    # this section will appear up in two sets in P corresponding to the two professors
    P={}
    for inst in ss.instructor.unique():
        P[inst]=set(ss[ss.instructor==inst].index)
    for inst in avai.keys():
        for pair in pairedProfs.keys():
            if inst in pair:
                if inst in P.keys():
                    P[inst].update(P[pair])
                else:
                    P[inst]=P[pair]
    for pair in pairedProfs.keys():
        P.pop(pair)

    # G : set of course groups that students usually take together.
    G=[]
    for i in groups.index:
        G.append(list(groups.loc[i].dropna()))

    # Cg: all the section combinations of course group g∈G
    C=[]
    for grp in G:
        tot_ss=[]
        for cl in grp:
            if cl in S.keys():
                tot_ss.append(S[cl])
        if len(tot_ss)>1:
            C.append(tot_ss)
    print('Input data ready:',pd.datetime.now()-start_time,flush=True)

    #setting up gurobi
    print(flush=True)
    print('-->>Start setting up scheduling optimization...',flush=True)
    start_time = pd.datetime.now()
    mod=Model()
    print('Model initiated:',pd.datetime.now()-start_time,flush=True)

    #set up decision variables
    start_time = pd.datetime.now()
    x=mod.addVars(I,J,K,vtype=GRB.BINARY)
    print('Variables set up:',pd.datetime.now()-start_time,flush=True)

    # Set up objective
    start_time = pd.datetime.now()
    mod.setObjective(sum(a.loc[i,j]*x[i,j,k] for k in K for i in I for j in J),
                     sense=GRB.MAXIMIZE)
    print('Objective set up:',pd.datetime.now()-start_time,flush=True)

    # Constraints
    ##1 Same course conflict
    start_time = pd.datetime.now()
    for course in S.values():
        if len(course)>1:
            for pair in D:
                mod.addConstr(sum(x[i,j,k] for k in K for i in course for j in pair)<=1)
            for j in U:
                mod.addConstr(sum(x[i,j,k] for k in K for i in course)<=1)
    print('Constraint #1 set up:',pd.datetime.now()-start_time,flush=True)

    ##2 Same classroom conflict
    start_time = pd.datetime.now()
    for k in K:
        for pair in D:
            mod.addConstr(sum(x[i,j,k] for i in I for j in pair)<=1)
        for j in U:
            mod.addConstr(sum(x[i,j,k] for i in I)<=1)
    print('Constraint #2 set up:',pd.datetime.now()-start_time,flush=True)

    ##3 Ineligible slots
    start_time = pd.datetime.now()
    for pair in A:
        i=pair[0]
        j=pair[1]
        for k in K:
            mod.addConstr(x[i,j,k]==0)
    print('Constraint #3 set up:',pd.datetime.now()-start_time,flush=True)

    ##4 Ineligible classrooms
    start_time = pd.datetime.now()
    for pair in H:
        i=pair[0]
        k=pair[1]
        for j in J:
            mod.addConstr(x[i,j,k]==0)
    print('Constraint #4 set up:',pd.datetime.now()-start_time,flush=True)

    ##5 Same section conflict
    start_time = pd.datetime.now()
    for i in I:
        mod.addConstr(sum(x[i,j,k] for j in J for k in K)<=1)
    print('Constraint #5 set up:',pd.datetime.now()-start_time,flush=True)

    ##6 Same professor conflict
    start_time = pd.datetime.now()
    for grp in P.values():
        if len(grp)>1:
            for pair in D:
                mod.addConstr(sum(x[i,j,k] for k in K for i in grp for j in pair)<=1)
            for j in U:
                mod.addConstr(sum(x[i,j,k] for k in K for i in grp)<=1)
    print('Constraint #6 set up:',pd.datetime.now()-start_time,flush=True)

    ##7 Together course conflict
    start_time = pd.datetime.now()
    ys={}
    for cg in C:
        # the cartesian product of all sets in cg
        # all the possible combinations of sections
        # when picking one section from each course in the group of courses
        # that students usually take together
        combos=list(it.product(*cg))
        # auxiliary variable that tracks whether any two sections in a combination
        # are assigned to the same slot or to overlapping time slots
        ys[str(cg)]=mod.addVars(combos,vtype=GRB.BINARY)
        for c in combos:
            # in rare cases, a combined section combined sections from two courses that students usually take together
            # because a combined section belongs to both mother courses
            # a combination can have duplicated sections (representing different courses)
            cc=set(c) # get rid of duplicates
            for pair in D:
                mod.addConstr(sum(x[i,j,k] for i in cc for k in K for j in pair)<=ys[str(cg)][c]*len(c)+1)
            for j in U:
                mod.addConstr(sum(x[i,j,k] for i in cc for k in K)<=ys[str(cg)][c]*len(c)+1)
        mod.addConstr(sum(ys[str(cg)][c] for c in combos)<=len(combos)-1)
    print('Constraint #7 set up:',pd.datetime.now()-start_time,flush=True)

    start_time = pd.datetime.now()
    print(flush=True)
    print('-->>Start scheduling...',flush=True)
    # set a time limit to stop Gurobi
    # mod.Params.TimeLimit = 30*60
    mod.optimize()
    print('Model optimized:',pd.datetime.now()-start_time,flush=True)
    print(flush=True)
    print('------------------------------------------------------------------',flush=True)
    print('------------------------------------------------------------------',flush=True)
    print(flush=True)
    print('Total preference score:',mod.objVal,flush=True)

    # Generate outputs
    # the output timetable is hard coded in this prototype, which can be updated
    hourList=[(datetime.combine(date.today(),time(8,0))+timedelta(minutes=30*i)).time() for i in range(27)]
    dayList=['M','T','W','H','F']
    timetables={}
    assigned=pd.DataFrame(columns=list(sections.columns)+\
    ['days','begin_time','end_time','classroom','capacity','preference_score'])
    unassigned=pd.DataFrame(columns=sections.columns)
    totAssignedLength=0 # tracks hours assigned
    for k in K:
        df=pd.DataFrame(index=hourList,columns=dayList)
        for j in J:
            for i in I:
                if '_' in i:
                    iList=i.split('_')
                else:
                    iList=[i]
                if x[i,j,k].x==1:
                    for ii in iList:
                        results=list(sections.loc[ii])
                        days=j[:-11]
                        begin_h=int(j[-10:-8])
                        begin_m=int(j[-7:-5])
                        begin_time=time(begin_h,begin_m)
                        length=int(sections.loc[ii,'length'])
                        end_time=(datetime.combine(date.today(),begin_time)+timedelta(minutes=length)).time()
                        capacity=cap.loc[k]
                        instructor=sections.loc[ii,'instructor']
                        # preference score is recalculated for each individual professor given their assigned time
                        # because previously some professors are combined
                        avaiDF=avai[instructor]
                        session=sections.loc[ii,'session']
                        if session in firstHalf:
                            halfList=[1]
                        elif session in secondHalf:
                            halfList=[2]
                        else:
                            halfList=[1,2]
                        unavai=False
                        tot_score=0
                        count=0
                        numHalf=math.ceil(length/30)
                        for n in range(numHalf):
                            start=(datetime.combine(date.today(),begin_time)+timedelta(minutes=30*n)).time()
                            for d in days:
                                if n==0:
                                    if '_' in i or session in full:
                                        df.loc[start,d]=i
                                    elif session in firstHalf:
                                        df.loc[start,d]=f'{i}_XXX'
                                    else:
                                        df.loc[start,d]=f'XXX_{i}'
                                else:
                                    df.loc[start,d]='----------'
                                for half in halfList:
                                    sscore=avaiDF.loc[str(start),f'{d}{half}']
                                    if sscore==0:
                                        unavai=True
                                        break
                                    else:
                                        tot_score+=sscore
                                        count+=1
                        if unavai==True:
                            score=0
                        else:
                            score=tot_score/count
                        results.extend([days,str(begin_time),str(end_time),k,capacity,score])
                        assigned.loc[ii]=results
                        timetables[k]=df
                        totAssignedLength+=numHalf*30*len(halfList)/2*len(days)
    for i in sections.index:
        if i not in assigned.index:
            unassigned.loc[i]=sections.loc[i]
    assigned=assigned[['department','course','session','instructor','preference_score',
                   'days','begin_time','end_time','classroom','seats_offered','capacity']].\
                   sort_values(['department','course']).reset_index().rename(columns={'index':'section'})
    unassigned=unassigned.sort_values(['department','course']).reset_index().rename(columns={'index':'section'})
    print('Total sections assigned: {} ({:.1f}% out of {})'.\
        format(len(assigned),len(assigned)/numSec*100,numSec),flush=True)
    print('Total length of sections assigned: {} hrs ({:.1f}% out of {})'.\
        format(totAssignedLength/60,totAssignedLength*100/60/totSecLength,totSecLength),flush=True)

    # write results to outputFile
    writer=pd.ExcelWriter(outputFile)
    assigned.to_excel(writer,sheet_name='AssignedSections',index=False)
    unassigned.to_excel(writer,sheet_name='UnassignedSections',index=False)
    for k in K:
        timetables[k].to_excel(writer,sheet_name=k)
    writer.save()
    print('Total time elapsed:', pd.datetime.now()-tot_start_time,flush=True)

if __name__=='__main__':
    import sys, os
    if len(sys.argv)!=3:
        print('Correct syntax: python scheduling.py inputFile outputFile')
    else:
        inputFile=sys.argv[1]
        outputFile=sys.argv[2]
        if os.path.exists(inputFile):
            optimize(inputFile,outputFile)
            print(f'Successfully optimized. Results in "{outputFile}"')
        else:
            print('Input file not Found!')
