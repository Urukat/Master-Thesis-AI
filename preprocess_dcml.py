import pandas as pd
import numpy as np
#import mozart_piano_sonatas.utils.feature_matrices as fm
from utils import notetype, load_dcml_tsv, name2tpc
#import os.path as path
from pathlib import Path
#from os import path, pardir
import tqdm
import multiprocessing as mp

# predefined values
# -----------------

# theoretical chord tones of the chord types in the corpus
# (pitch class is given in fifths)
chord_types = {
    'M': [0,1,4],
    'm': [0,1,-3],
    'o': [0,-3,-6],
    '+': [0,4,8],
    'Mm7': [0,1,4,-2],
    'mm7': [0,1,-3,-2],
    'MM7': [0,1,4,5],
    'mM7': [0,1,-3,5],
    '%7': [0,-3,-6,-2],
    'o7': [0,-3,-6,-9],
    '+7': [0,4,8,-2],
    'It':  [0, -10, -6],     # (F#, Ab, C)     over F# = (P1, d3, d5)
    'Fr':  [0, 4,   -6, -2], # (D, F#, Ab, C)  over D  = (P1, M3, d5, m7)
    'Ger': [0, -10, -6, -9], # (F#, Ab, C, Eb) over F# = (P1, d3, d5, d7)
}

# corpus directory

# helper function
# ---------------

def total_onsets(events, measures):
    """
    Returns the onsets of events (chord labels or notes) relative to the beginning of the piece
    by converting them from measure-relative notation
    """
    moffsets = measures.act_dur.values.cumsum()
    monsets = moffsets - measures.act_dur.values
    mi = events.mc - 1
    return events.mc_onset + monsets[mi]

def merge_ties(notes):
    """
    Returns a copy of notes with ties merged.
    """
    notes = notes.copy()
    beginnings = notes[notes.tied == 1]
    continues = notes[notes.tied < 1]
    for i in beginnings.index:
        on = notes.total_onset[i]
        off = notes.total_offset[i]
        midi = notes.midi[i]
        tpc = notes.tpc[i]
        while True:
            cont = continues[(continues.total_onset == off) &
                             (continues.midi == midi) &
                             (continues.tpc == tpc)].first_valid_index()
            if cont is None:
                break
            off = continues.total_offset[cont]
            if continues.tied[cont] == -1:
                break
        notes.at[i, 'total_offset'] = off
        notes.at[i, 'duration'] = off - on
    return notes[~(notes.tied < 1).fillna(False)]

def load_dfs(corpus, piece):
    """
    Loads and preprocesses dataframes for the notes and chord labels of a piece.
    """
    measures = load_dcml_tsv(corpus, piece, 'measures')
    
    notes = load_dcml_tsv(corpus, piece, 'notes')
    notes['total_onset'] = total_onsets(notes, measures)
    notes['total_offset'] = notes.total_onset.values + notes.duration.values
    notes = merge_ties(notes)
    max_offset = notes.total_offset.values.max()
    
    harmonies = load_dcml_tsv(corpus, piece, 'harmonies')
    harmonies = harmonies[~harmonies.chord.isnull()]
    harmonies['total_onset'] = total_onsets(harmonies, measures)
    harmonies['total_offset'] = np.append(harmonies.total_onset.values[1:], max_offset)
    if 'special' in harmonies.columns:
        harmonies['actual_chord_type'] = harmonies.special.fillna(harmonies.chord_type)
    else:
        harmonies['actual_chord_type'] = harmonies.chord_type
    
    return notes, harmonies

# extracting chords
# -----------------

# def get_chords(notes, harmonies):
#     """
#     Computes chords as label x note pairs for a piece (given by its notes and chord labels).
#     Pairs that belong to the same chord get the same id, starting from id_offset.
#     Returns the dataframe of chords and the highest used id.
#     """
#     key = name2tpc(harmonies.globalkey[0])
#
#     # setup the columns of the result dataframe
#     chordids = np.empty(0, dtype=int)
#     labels   = np.empty(0, dtype=str)
#     fifths   = np.empty(0, dtype=int)
#     types    = np.empty(0, dtype=str)
#     # running id counter
#     current_id = 0
#
#     # for checking whether the chord label is empty at some point
#     chord_is_null = harmonies.chord.isnull()
#
#     # iterate over all harmonies
#     for i, ih in enumerate(harmonies.index):
#         # chord label empty or '@none'? then skip
#         if chord_is_null[ih] or harmonies.chord[ih] == '@none':
#             continue
#
#         # get info about the current harmony
#         on = harmonies.total_onset[ih]
#         off = harmonies.total_offset[ih]
#         label = harmonies.actual_chord_type[ih]
#         root = harmonies.root[ih] + key
#
#         # compute the corresponding notes, their pitches, and their note types
#         inotes = (notes.total_offset > on) & (notes.total_onset < off)
#         pitches = notes.tpc[inotes].values - root
#         chord_tones = chord_types[label]
#         note_types = [notetype(p, pitches, chord_tones) for p in pitches]
#         if(len(pitches) == 0):
#             continue
#
#         # add everything to the dataframe columns
#         chordids = np.append(chordids, np.repeat(current_id, len(pitches)))
#         labels   = np.append(labels, np.repeat(label, len(pitches)))
#         fifths   = np.append(fifths, pitches)
#         types    = np.append(types, note_types)
#         current_id += 1
#
#     # create the result dataframe
#     chords_df = pd.DataFrame({
#         'chordid': chordids,
#         'label': labels,
#         'fifth': fifths,
#         'type': types})
#     return chords_df, current_id # current_id = number of chords

def get_chords(notes, harmonies, filename):
    """
    Computes chords as label x note pairs for a piece (given by its notes and chord labels).
    Pairs that belong to the same chord get the same id, starting from id_offset.
    Returns the dataframe of chords and the highest used id.
    Now includes 'onset' and 'filename' information for each chord.
    """
    key = name2tpc(harmonies.globalkey[0])

    chordids = np.empty(0, dtype=int)
    labels = np.empty(0, dtype=str)
    fifths = np.empty(0, dtype=int)
    types = np.empty(0, dtype=str)
    onsets = np.empty(0, dtype=float)
    filenames = np.empty(0, dtype=str)
    current_id = 0

    chord_is_null = harmonies.chord.isnull()

    for i, ih in enumerate(harmonies.index):
        if chord_is_null[ih] or harmonies.chord[ih] == '@none':
            continue

        on = harmonies.total_onset[ih]
        off = harmonies.total_offset[ih]
        label = harmonies.actual_chord_type[ih]
        root = harmonies.root[ih] + key

        inotes = (notes.total_offset > on) & (notes.total_onset < off)
        pitches = notes.tpc[inotes].values - root
        chord_tones = chord_types[label]
        note_types = [notetype(p, pitches, chord_tones) for p in pitches]
        if len(pitches) == 0:
            continue

        chordids = np.append(chordids, np.repeat(current_id, len(pitches)))
        labels = np.append(labels, np.repeat(label, len(pitches)))
        fifths = np.append(fifths, pitches)
        types = np.append(types, note_types)
        onsets = np.append(onsets, np.repeat(on, len(pitches)))
        filenames = np.append(filenames, np.repeat(filename, len(pitches)))
        current_id += 1

    chords_df = pd.DataFrame({
        'chordid': chordids,
        'label': labels,
        'fifth': fifths,
        'type': types,
        'onset': onsets,
        'filename': filenames
    })
    return chords_df, current_id


# processing files
# ----------------

def get_chords_from_piece(piece):
    """
    Same as get_chords, but takes a corpus subdirectory and a piece id.
    Includes filename in the information passed to get_chords.
    """
    folder, file = piece
    name = f"{folder} {file}"
    try:
        notes, harmonies = load_dfs(folder, file)
        chords, n_chords = get_chords(notes, harmonies, name)
        return chords, n_chords, name
    except FileNotFoundError:
        print(f'file not found for {name}')
        return None
    except ValueError:
        print(f'ValueError in {name}')
        return None
    except (KeyboardInterrupt):
        print("interrupted by user, exiting.")
        quit()
    except Exception as e:
        print(f'error while processing {name}:\n{e}')
        return None

def get_chords_from_files(filelist):
    """
    Returns the combined chords for several pieces.
    Takes a list of subdirectory x piece pairs.
    """

    # load all files and extract chords (slow, parallelized)
    with mp.Pool() as pool:
        outputs = list(tqdm.tqdm(pool.imap(get_chords_from_piece, filelist), total = len(filelist)))

    # collect results in one large dataframe (fast)
    all_chords = pd.DataFrame()
    files = []
    total_chords = 0
    for output in tqdm.tqdm(outputs):
        if output is not None:
            chords, n_chords, name = output
            chords['chordid'] += total_chords
            all_chords = pd.concat((all_chords, chords))
            total_chords += n_chords
            files.append(name)

    # write log
    print(f"got {total_chords} chords and {len(all_chords)} notes from the {len(files)} files listed in data/preprocess_dcml.txt")
    with open(Path("data", "preprocess_dcml.txt"),"w") as f:
      print(f"got {total_chords} chords and {len(all_chords)} notes from the following {len(files)} files",file=f)
      f.write("\n".join(files))

    return all_chords.reset_index(drop=True)

def get_corpus_pieces(corpus):
    """
    Returns a list of pieces in a corpus as subdirectory x piece pairs.
    """
    corpus = Path(corpus)
    print("fetching pieces from", corpus)
    dirs = [d.parent for d in corpus.glob('*/harmonies')]
    print(dirs)
    # sort for consistent order
    files = sorted((d, f.stem)
                   for d in dirs
                   for f in d.glob('harmonies/*.tsv'))
    return files

# script
# ------

if __name__ == "__main__":
    print("scanning corpus...")
    pieces_dcml = get_corpus_pieces(Path("data", "dcml_corpora"))
    pieces_romantic = get_corpus_pieces(Path("data", "romantic_piano_corpus"))
    pieces = pieces_dcml + pieces_romantic
    print(len(pieces), "pieces")
    print("extracting chords from pieces...")
    all_chords = get_chords_from_files(pieces)
    print("writing chords...")
    all_chords.to_csv(Path('data', 'dcml2.tsv'), sep='\t', index=False)
    print("done.")
