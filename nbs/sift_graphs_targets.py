## Get Roi in model for now toy images 
import fire
import os
import lmdb
import cv2
import numpy as np
import six
import matplotlib 
import math
from PIL import Image
from matplotlib import pyplot as plt
import lmdb
from tqdm import tqdm 
# def get_rois(img_dir, env, key ,lmdbb=True, cv=False, show=False):
#     if cv :
#         demo_imgs_name = os.listdir(img_dir)
#         demo_imgs = []
#         for nm in demo_imgs_name :
#             demo_imgs.append(cv2.imread(os.path.join(img_dir, nm)))

#         if show :     
#             for img in demo_imgs:
#                 plt.imshow(img)
#                 plt.show()

#         return demo_imgs
    
#     if lmdbb : 
      
        
#                 image_arr.append((img_key.decode(), img))
#         return image_arr


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v.encode())    

def get_kp(channel, show=False):
    # print('doing sift')
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(channel, None)
    min_resp = 99999
    max_resp = 0
    kp_filtered = [] 
    # print("kp vs filter", len(kp), len(kp_filtered), kp)

    for k in kp : 
        min_resp = k.response if k.response < min_resp else min_resp
        max_resp = k.response if k.response > max_resp else max_resp

    resp_filter = 0.6 * (max_resp - min_resp)
    for k in kp:
        if k.response >= resp_filter : 
            kp_filtered.append(k)
    kp_filtered, desc = sift.compute(channel, kp_filtered)    
    if show:
        print("kp vs filter", len(kp), len(kp_filtered))

    return kp, kp_filtered, desc

def make_sift(img, show=False):
    # gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = np.float32(gray)
    # dst = cv2.cornerHarris(gray,2,3,0.04)
    # img[dst>0.01*dst.max()]=[255,0,255]
    # plt.imshow(img)
    # plt.show()
    # # # return(None)
    # print(img)
    # print(np.shape(img))

    im_grey = img.copy()
    kp = []; 
    kp_filtered = []; 
    desc = []
    kpc1 = get_kp(img[:,:,0])
    kpc2 = get_kp(img[:,:,1])
    kpc3 = get_kp(img[:,:,2])
    
    if kpc1[0]:
        for k in kpc1[0] : 
            kp.append(k)
        for k in kpc1[1] : 
            kp_filtered.append(k)
        for k in kpc1[2] : 
            desc.append(k)
    if kpc2[0]:
        for k in kpc2[0] : 
            kp.append(k)
        for k in kpc2[1] : 
            kp_filtered.append(k)
        for k in kpc2[2] : 
            desc.append(k)
    if kpc3[0]:
        for k in kpc3[0] : 
            kp.append(k)
        for k in kpc3[1] : 
            kp_filtered.append(k)
        for k in kpc3[2] : 
            desc.append(k)

    if show:
        print("ksp", np.shape(kp),"after filter",  np.shape(kp_filtered))
        print("desc", np.shape(desc))
        tmp_img1 = cv2.drawKeypoints(im_grey, kp, im_grey)
        f, axarr = plt.subplots(2,1)
        axarr[0].imshow(tmp_img1) 
        tmp_img2= cv2.drawKeypoints(img, kp_filtered, img)
        axarr[1].imshow(tmp_img2)
        plt.show()

    
    return kp_filtered, desc
def generate_graph(nodes, img, show=False):
    # only doing undirected graphs for now. 
    edges = []
    for node in nodes :
        for node2 in nodes:
            if is_edge(nodes, node, node2):
                if (node2,node) in edges : 
                    pass
                else:   
                    edges.append((node, node2))
                    # edges.append((nodes[node][2], nodes[node2][2]))
            else:
                pass
    
    if show:
        for node in nodes:
            cv2.circle(img, (nodes[node][0],nodes[node][1]), 1, (0,0,255))
        for ed in edges:
            cv2.line(img, (nodes[ed[0]][0],nodes[ed[0]][1]), (nodes[ed[1]][0], nodes[ed[1]][1]),(255,0,0),1)

        plt.imshow(img)
        plt.show()

    return edges

def is_edge(nodes, a, b, dthreshold=100, nthreshold=2):
    # print(a, b)
    pta = (nodes[a][0], nodes[a][1])
    ptb = (nodes[b][0], nodes[b][1])
    node_dist =  np.linalg.norm(np.array(a)-np.array(b))
    dist =  np.linalg.norm(np.array(pta)-np.array(ptb))
    # dist =  math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 )
    isit =  True if dist<=dthreshold and dist > 0 and node_dist<=nthreshold  else  False
    # print("is edge", dist, isit)
    return isit
    
def save_graph_text(graph, dir_to_sav=None):
    ## Direct write to file method - vv slow
    # # print("in save graphs", graphs[0])
    # kol = os.path.join(dir_to_sav, "graphs", graphs[3])
    # # print("Saving graph txt to", kol)
    # # pickle.dump( graphs, open( kol, "wb" ) )
    # with open(kol, 'w') as f : 
    #     for e in graphs[1]:
    #         ln = str(e[0]) + " 0 "+ str(e[1]) 
    #         f.write(ln)
    #         f.write('\n')

    # f.close()
    # kol = os.path.join(dir_to_sav, "nodes", graphs[3])
    # # print("Saving nodes txt to", kol)
    # # pickle.dump( graphs, open( kol, "wb" ) )
    # with open(kol, 'w') as f : 
    #     for key in graphs[0]:
    #         line_to_write = "" + str(key)+ " " + str(graphs[0][key][2])+ " " + str(graphs[0][key][3])+ " "
    #         for a in graphs[2][key]:
    #             line_to_write += str(a)+" "
    #         f.write(line_to_write)
    #         f.write('\n')
    # f.close()
    
    ## Return serializd graph to save to lmdb
    nodes, edges, descs = graph
    # print("graph[0]", nodes)
    # print("graph[1]", edges)
    # print("graph[2]", descs)

    g_str = ""
    for ed in edges:
        g_str += str(ed) + " "
    # print("gstring \t", g_str)
    n_str = ""

    for nd in nodes:
        n_str +=str(nd)+' '+ str(nodes[nd][2]) + " " + str(nodes[nd][3]) + " " 
        for des in descs[nd]:
            n_str += str(des) + " "

        n_str+=";"    
    # print("n_str \t", n_str)

    return(g_str, n_str)    

    
def get_nd_tp(angle):
    if angle>=0 & angle<=44:
        return 0 
    elif angle>=45 & angle<=89:
        return 1 
    elif angle>=90 & angle<=134:
        return 2 
    elif angle>=135 & angle<=179:
        return 3 
    elif angle>=180 & angle<=224:
        return 4 
    elif angle>=225 & angle<=269:
        return 5 
    elif angle>=270 & angle<=314:
        return 6 
    elif angle>=315 & angle<=359:
        return 7



def main():
    # img_dir = "/home/sahmed9/Documents/reps/deep-text-recognition-benchmark/"
    # img_dir = "/home/sahmed9/Documents/data/kenny_data_share/images_png/aa/"
    data_root = "/run/user/1001/gvfs/sftp:host=alice.cedar.buffalo.edu/data/lmdb/data_lmdb_release/training/MJ"
    # folder = ["MJ_test", "MJ_train", "MJ_valid"]
    folder = ["MJ_valid/MJ_valid_label"]
    folder = ["MJ_train/MJ_train_label"]
    po = 0
    for diri in folder :
        graphs_cache = {}
        img_dir = os.path.join(data_root, diri)
        os.makedirs(os.path.join("./", diri), exist_ok=True)
        outputPath = os.path.join("./", diri)
        env2 = lmdb.open(outputPath, map_size=1099511627776)
        file_len = 37000
        print("Loading from dir", img_dir)
        # exit()
        # file_len = 0 
        # s = os.listdir(img_dir)
        # s.sort()
        # file_len = len(s) 
        # print("Total", file_len, "from", s[0], "to", s[-1])

        for i in tqdm(range(1, file_len)):
            try:
                img_key = 'image-%09d'.encode() % i
                label_key = 'label-%09d' % i + '.png'
                graph_key = 'graph-%09d'.encode() % i
                node_key = 'nodes-%09d'.encode() % i

                img = cv2.imread(os.path.join(img_dir, label_key))
                imgh, imgw , _ =  np.shape(img)
                # print("img shape", np.shape(img))

                if imgh * imgw <100 :
                    print("\nafter buffer", type(img), np.shape(img))
                    pass
                else :
                    kps, descs = make_sift(img)
                    # kps.append(kp)
                    # descs.append(desc)
                    nodes = {}
                    for i1, kp in enumerate(kps) :
                        # node_type = get_nd_tp(kp.angle) 
                        # print(kp.pt[0], kp.pt[1], kp.angle, kp.size)
                        # nodes[i1] = (kp.pt[0], kp.pt[1], kp.angle, kp.size)
                        nodes[i1] = (round(kp.pt[0]), round(kp.pt[1]), kp.angle, kp.size)
                        # nodes[i1] = (kp.pt[0], kp.pt[1], node_type)
                        # nodes[i1] = (round(kp.pt[0]), round(kp.pt[1]), node_type)
                    # print("NODES", len(nodes))    
                    edges = generate_graph(nodes, img)
                    # print("Edges", len(edges))
                    graph = (nodes, edges, descs)   
                    graphs_cache[graph_key], graphs_cache[node_key]  = save_graph_text(graph)
                    # print(graphs_cache)
                    # exit()
                    try:
                        if i % 1000 == 0:
                            writeCache(env2, graphs_cache)
                            # exit()
                            graphs_cache = {}
                            print('Written %d / %d' % (i, file_len))
                        # graphs.append((nodes, edges, descs, im_nm))
                    except Exception as e:
                        print("exception occured in writeCache", e)
                        print("cache", len(graphs_cache))
                        continue
                    # print("DONE EXIT NOW") 
                    # exit()
    
            except Exception as e :
                print("exception occured in sift/generate_graph", e)
                print("cache", len(graphs_cache))
                continue 
            
        # print("First graph , nodes, edges, descs", len(graphs[0][0]), len(graphs[0][1]), len(graphs[0][2]))
        # print("Total graphs ", len(graphs))


if __name__ == '__main__':
    main()