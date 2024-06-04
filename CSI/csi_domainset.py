import itertools
import numpy as np
import pandas as pd
from itertools import combinations

widar_sub=90
dataset_csi_size={
            'Widar3':{
                'amp':(widar_sub, 2500),
                'pha':(widar_sub, 2500),
                'amp+pha':(widar_sub*2, 2500),
                'img':(225,225),
                }
        }


def get_domains(dataset_type,domain_type,ibegin,imax,rxs=None,oris=None,loc_ids=None):
    dataset_domain_list={}
    if dataset_type=='Widar3':
        if domain_type in ['loc']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=rxs,oris=oris)
        elif domain_type in ['ori']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=rxs,loc_ids=loc_ids)
        elif domain_type in ['rx']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,loc_ids=loc_ids,oris=oris)
        elif domain_type in ['user']:
            dataset_domain_list=get_widar3_all_domains(ibegin,imax,domain_type,dataset_domain_list,rxs=rxs,oris=oris,loc_ids=loc_ids)
        else:
            raise ValueError('wrong')
    return dataset_domain_list



def get_widar3_all_domains(ibegin,imax,domain_type,domain_list,rxs=None,oris=None,loc_ids=None):
    i=0
    domain_list['Widar3']=[]
    if loc_ids is None:
        loc_ids=['1','2','3','4','5']
    else:
        loc_ids=loc_ids

    #方向上可以不考虑，天线为组合特征

    if rxs is None:
        rx_ids=['1','2','3','4','5','6']
    else:
        rx_ids=rxs

    if oris is None:
        ori_ids = ['1', '2', '3', '4', '5']
    else:
        ori_ids=oris


    if domain_type=='room':
        user='3'
        rooms=['1','2','3']
        for ori in ori_ids:
            for rx in rx_ids:
                for loc in loc_ids:
                    source_rooms=list(itertools.combinations(rooms, 2))
                    for source_room in source_rooms:
                        i=i+1
                        target_room=set(rooms).difference(set(source_room))
                        source_damains=[]
                        target_damains=[]
                        for room in source_room:
                            source_domain_name='room_'+room+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                            source_damains.append(source_domain_name)
                        for room in target_room:
                            target_domain_name='room_'+room+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                            target_damains.append(target_domain_name)
                        dic={
                            'exp_type':'cross_'+domain_type+str(i), 
                            'source_domains':source_damains, 
                            'target_domains':target_damains,
                            'n_subdomains':{'room':len(source_room)},
                            }
                        if i>=ibegin:
                            domain_list['Widar3'].append(dic)
                        
                        if i>=imax:
                            return domain_list

    if domain_type=='ori':
        widar_room_user_ids={
            1:['1','2','3']
            }  
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for rx in rx_ids:
                    for user in widar_room_user_ids[room]:
                        source_oris=list(itertools.combinations(ori_ids, 4))
                        for source_ori in source_oris:
                            i=i+1
                            target_ori=set(ori_ids).difference(set(source_ori))
                            source_damains=[]
                            target_damains=[]
                            for ori in source_ori:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for ori in target_ori:
                                target_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'ori':len(source_ori)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list
    if domain_type=='loc': 
        widar_room_user_ids={
            1:['1','2','3'] #3-9 *
        }
        for ori in ori_ids:
            for rx in rx_ids:
                for room in widar_room_user_ids:
                    for user in widar_room_user_ids[room]:
                        source_locs=list(itertools.combinations(loc_ids, 4))
                        for source_loc in source_locs:
                            i=i+1
                            target_loc=set(loc_ids).difference(set(source_loc))
                            source_damains=[]
                            target_damains=[]
                            for loc in source_loc:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for loc in target_loc:
                                target_domain_name = 'room_' + str(
                                    room) + '_user_' + user + '_loc_' + loc + '_ori_' + ori + '_rx_' + rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'loc':len(source_loc)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list
    if domain_type=='user':
        widar_room_user_ids={
            # 1:['1','2','3','5','10','11'] #3-9 *
            3:['3','7','8','9'] #3-9 *
            }
        for loc in loc_ids:
            for ori in ori_ids:
                for rx in rx_ids:
                    for room in widar_room_user_ids:
                        source_users=list(itertools.combinations(widar_room_user_ids[room], len(widar_room_user_ids[room])-1))
                        for source_user in source_users:
                            i=i+1
                            target_user=set(widar_room_user_ids[room]).difference(set(source_user))
                            source_damains=[]
                            target_damains=[]
                            for user in source_user:
                                source_domain_name='room_'+str(room)+'_user_'+user+'_loc_'+loc+'_ori_'+ori+'_rx_'+rx
                                source_damains.append(source_domain_name)
                            for user in target_user:
                                target_domain_name = 'room_' + str(
                                    room) + '_user_' + user + '_loc_' + loc + '_ori_' + ori + '_rx_' + rx
                                target_damains.append(target_domain_name)
                            dic={
                                'exp_type':'cross_'+domain_type+str(i), 
                                'source_domains':source_damains, 
                                'target_domains':target_damains,
                                'n_subdomains':{'user':len(source_user)},
                                }
                            if i>=ibegin:
                                domain_list['Widar3'].append(dic)
                            
                            if i>=imax:
                                return domain_list
    if domain_type == 'rx':
        widar_room_user_ids = {
            # 1:['1','2','3'] #3-9 *
            1: ['1', '2', '3', ]
        }
        for room in widar_room_user_ids:
            for loc in loc_ids:
                for ori in ori_ids:
                    for user in widar_room_user_ids[room]:
                        source_rxs = list(itertools.combinations(rx_ids, 2))
                        for source_rx in source_rxs:
                            i = i + 1
                            target_rx = set(rx_ids).difference(set(source_rx))
                            source_damains = []
                            target_damains = []
                            for rx in source_rx:
                                source_domain_name = 'room_' + str(
                                    room) + '_user_' + user + '_loc_' + loc + '_ori_' + ori + '_rx_' + rx
                                source_damains.append(source_domain_name)
                            for rx in target_rx:
                                target_domain_name = 'room_' + str(
                                    room) + '_user_' + user + '_loc_' + loc + '_ori_' + ori + '_rx_' + rx
                                target_damains.append(target_domain_name)
                            dic = {
                                'exp_type': 'cross_' + domain_type + str(i),
                                'source_domains': source_damains,
                                'target_domains': target_damains,
                                'n_subdomains': {'rx': len(source_rx)},
                            }
                            if i >= ibegin:
                                domain_list['Widar3'].append(dic)

                            if i >= imax:
                                return domain_list
    return domain_list






