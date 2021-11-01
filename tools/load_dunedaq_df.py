#!/usr/bin/env python

# Import DUNEDAQ classes
import hdf5libs
import daqdataformats
import detdataformats
import detchannelmaps

# Import utilities
import rich
import numpy as np
import pandas as pd
import scipy as sp
import collections

from rich.console import Console
from rich.panel import Panel



def main(console: Console, raw_file: str, trg_n: int, max_tr: int=100):

    ch_map = detchannelmaps.make_map('VDColdboxChannelMap')

    dd = hdf5libs.DAQDecoder(raw_file, 100)

    datasets = dd.get_datasets()

    tr_datasets = [ d for d in datasets if d.startswith(f'//TriggerRecord{trg_n:05}') and 'TriggerRecordHeader' not in d]

    console.print(f"{len(tr_datasets)} found: {tr_datasets}")

    dfs = []
    for d in tr_datasets:
        console.print(f"Inspecting {d}")
        frag = dd.get_frag_ptr(d)
        frag_hdr = frag.get_header()

        console.print(f"Run number : {frag.get_run_number()}")
        console.print(f"Trigger number : {frag.get_trigger_number()}")
        console.print(f"Trigger TS    : {frag.get_trigger_timestamp()}")
        console.print(f"Window begin  : {frag.get_window_begin()}")
        console.print(f"Window end    : {frag.get_window_end()}")
        console.print(f"Fragment type : {frag.get_fragment_type()}")
        console.print(f"Fragment code : {frag.get_fragment_type_code()}")
        console.print(f"Size          : {frag.get_size()}")

        n_frames = (frag.get_size()-frag_hdr.sizeof())//detdataformats.WIBFrame.sizeof()
        console.print(f"Number of WIB frames: {n_frames}")

        wf = detdataformats.WIBFrame(frag.get_data())
        wh = wf.get_wib_header()

        console.print(f"crate: {wh.crate_no}, slot: {wh.slot_no}, fibre: {wh.fiber_no}")
        crate_no = wh.crate_no 
        slot_no = wh.slot_no
        fiber_no = wh.fiber_no if not slot_no in [2,3] else {1:2, 2:1}[fiber_no]
        off_chans = [ch_map.get_offline_channel_from_crate_slot_fiber_chan(crate_no, slot_no, fiber_no, c) for c in range(256)]

        ts = np.zeros(n_frames, dtype='uint64')
        adcs = np.zeros(n_frames, dtype=('uint16', 256))

        for i in range(n_frames):
            if i%1000  == 0:
                print(i)
            wf = detdataformats.WIBFrame(frag.get_data(i*detdataformats.WIBFrame.sizeof())) 
            ts[i] = wf.get_timestamp()
            adcs[i] = [wf.get_channel(c) for c in range(256)]
            
        df = pd.DataFrame(collections.OrderedDict([('ts', ts)]+[(f'{off_chans[c]:04}', adcs[:,c]) for c in range(256)]))
        df = df.set_index('ts')
        console.print(df)

        dfs.append(df)

    tr_df = pd.concat(dfs, axis=1)
    tr_df = tr_df.reindex(sorted(tr_df.columns), axis=1)
    return tr_df


if __name__ == '__main__':
    console = Console()
    
    # rawfile='np02_bde_coldbox_run011889_0000_20211027T171341.hdf5'
    raw_file='np02_bde_coldbox_run011918_0002_20211029T122926.hdf5'
    trg_n = 336
    dd = hdf5libs.DAQDecoder(raw_file, 100)

    # datasets = dd.get_datasets()
    # console.print(f"{len(datasets)} found: {datasets[:10]}...")

    df = main(console, raw_file, trg_n)