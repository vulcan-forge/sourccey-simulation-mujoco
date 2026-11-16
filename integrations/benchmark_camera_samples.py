"""Read local benchmark videos and paired telemetry for camera inspection.

Run with the existing LeRobot Python (PyAV, PyArrow, Pillow). Sources are read
only; frame samples and their provenance are written under the output path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image,ImageDraw

CAMERAS=('front_left','front_right','wrist_left','wrist_right')


def extract(source,output):
    source,output=Path(source),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for root in sorted(source.iterdir()):
        if not (root/'meta/info.json').is_file():
            continue
        info=json.loads((root/'meta/info.json').read_text())
        data=pq.read_table(sorted((root/'data').glob('*/*.parquet')))
        episodes=pq.read_table(sorted((root/'meta/episodes').glob('*/*.parquet'))).to_pylist()
        rows=data.to_pylist()
        selected=sorted(set([0,min(300,len(rows)-1),len(rows)//2,len(rows)-1]))
        directory=output/root.name
        directory.mkdir(exist_ok=True)
        sheet=Image.new('RGB',(1280,262*len(selected)),(20,23,27))
        draw=ImageDraw.Draw(sheet)
        for selection,index in enumerate(selected):
            row=rows[index]
            ep=next(e for e in episodes if e['episode_index']==row['episode_index'])
            case=dict(dataset=root.name,frame=index,episode=row['episode_index'],
                timestamp=float(row['timestamp']),state=dict(zip(info['features']['observation.state']['names'],row['observation.state'])),images={})
            for column,camera in enumerate(CAMERAS):
                key='videos/observation.images.'+camera
                video=root/'videos'/('observation.images.'+camera)/f'chunk-{ep[key+"/chunk_index"]:03d}'/f'file-{ep[key+"/file_index"]:03d}.mp4'
                target=ep[key+'/from_timestamp']+float(row['timestamp'])
                with av.open(str(video)) as container:
                    stream=container.streams.video[0]
                    container.seek(int(target/stream.time_base),stream=stream,backward=True)
                    nearest=None
                    for frame in container.decode(video=0):
                        if nearest is None or abs(float(frame.time)-target)<abs(float(nearest.time)-target):
                            nearest=frame
                        if float(frame.time)>=target:
                            break
                    if nearest is None or abs(float(nearest.time)-target)>.05:
                        raise ValueError(f'No synchronized video at {video}: {target}')
                    image=nearest.to_image()
                    filename=f'{index:05d}_{camera}.png'
                    image.save(directory/filename)
                    case['images'][camera]=dict(path=str((directory/filename).resolve()),video=str(video),pts_s=float(nearest.time),target_s=target)
                    sheet.paste(image,(column*320,selection*262+22))
                    draw.text((column*320+5,selection*262+4),f'{camera} | frame {index} | {row["timestamp"]:.1f}s',fill='white')
            cases.append(case)
        sheet.save(directory/'samples.png')
        print(f'{root.name}: {len(rows)} frames, {len(episodes)} episodes; sampled {selected}',flush=True)
    manifest=dict(source=str(source),cameras=list(CAMERAS),cases=cases,source_modified=False,
        note='Observed state paired to nearest video PTS within 50 ms; no physical calibration inferred.')
    (output/'samples.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    visual_constraints(output)
    return manifest


def visual_constraints(output):
    """Approximate image constraints, separated from physical calibration."""
    import cv2
    cv2.setRNGSeed(0)
    output=Path(output)
    pairs=[]
    directories=sorted(output.glob('sourccey-*'))
    for directory in directories:
        a=cv2.imread(str(directory/'00000_front_left.png'))
        b=cv2.imread(str(directory/'00000_front_right.png'))
        mask=np.zeros((240,320),dtype=np.uint8);mask[:85]=255
        sift=cv2.SIFT_create(contrastThreshold=.015)
        ka,da=sift.detectAndCompute(a,mask);kb,db=sift.detectAndCompute(b,mask)
        if da is None or db is None:
            continue
        matched=cv2.BFMatcher().knnMatch(da,db,k=2)
        matches=[m for m,n in matched if m.distance<.70*n.distance]
        if len(matches)<5:
            continue
        pa=np.float32([ka[m.queryIdx].pt for m in matches]);pb=np.float32([kb[m.trainIdx].pt for m in matches])
        _,valid=cv2.findHomography(pa,pb,cv2.RANSAC,2)
        good=[m for m,v in zip(matches,valid.ravel()) if v]
        pairs.extend(dict(left=ka[m.queryIdx].pt,right=kb[m.trainIdx].pt,dataset=directory.name) for m in good)
        cv2.imwrite(str(directory/'stereo_matches.png'),cv2.drawMatches(a,ka,b,kb,good,None,flags=2))
    (output/'stereo_points.json').write_text(json.dumps(pairs,indent=2)+'\n')
    contours={}
    for camera in ('wrist_left','wrist_right'):
        rows=[]
        for directory in directories:
            image=cv2.imread(str(directory/f'00000_{camera}.png'))
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            mask=cv2.morphologyEx((gray<90).astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
            curve=[]
            for x in range(10,311,10):
                values=np.flatnonzero(mask[90:,x])
                curve.append(int(values[0]+90) if len(values) else 239)
            rows.append(curve)
        contours[camera]=dict(x=list(range(10,311,10)),y=np.median(rows,axis=0).tolist(),per_dataset=rows)
    (output/'palm_contours.json').write_text(json.dumps(contours,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path('artifacts/benchmark_camera_fit'))
    args=parser.parse_args()
    extract(args.source,args.output)
