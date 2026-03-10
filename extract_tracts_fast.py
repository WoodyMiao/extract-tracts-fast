#!/usr/bin/env python

# Adapted for speed from https://github.com/Atkinson-Lab/Tractor/blob/master/scripts/extract_tracts.py
# Support at most 10 ancestries (coded as 0-9) in a MSP file
# Key optimizations:
# - Vectorized computation using numpy arrays and broadcasting (cannot handle missing values)
# - Parallelized file writing using queue-based threading (one thread per output file)
# - Removed VCF output functionality (only dosage and hapcount files)
# - Selectable output compression: gzip, pgzip (parallel gzip), or plain text

import argparse
import gzip
import logging
import numpy as np
import threading
import queue

# Configure logging with improved format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--vcf", required=True, help="Path to phased genotypes, compressed VCF file (*.vcf.gz) or BCF file (*.bcf)")
parser.add_argument("--msp", required=True, help="Path to ancestry calls, MSP file (*.msp or *.msp.tsv)")
parser.add_argument("--num-ancs", type=int, required=True, help="Number of ancestral populations within the VCF file.")
parser.add_argument("--out-pre", required=True, help="Output prefix for all output files.")
parser.add_argument("--compress", choices=["gzip", "pgzip", "txt"], default="gzip",
                    help="Output compression format: 'gzip' (default, standard gzip), "
                         "'pgzip' (parallel gzip, requires pgzip package), "
                         "or 'txt' (plain text, no compression). "
                         "Output file extensions will be .txt.gz for gzip/pgzip, or .txt for plain text.")
parser.add_argument("--pgzip-threads", type=int, default=2,
                    help="Number of compression threads per output file when using --compress pgzip (default: 2).")
parser.add_argument("--queue-size", type=int, default=1000,
                    help="Maximum number of lines to buffer in each write queue (default: 1000). "
                         "Increase if writer threads can keep up; decrease to reduce memory usage.")

args = parser.parse_args()

logger.info("Input VCF file: %s", args.vcf)
logger.info("Input MSP file: %s", args.msp)
logger.info("Number of ancestries: %d", args.num_ancs)
logger.info("Output prefix: %s", args.out_pre)
logger.info("Compression mode: %s", args.compress)

# Set up compression backend
if args.compress == "pgzip":
    try:
        import pgzip
        open_func = lambda path: pgzip.open(path, "wt", thread=args.pgzip_threads)
        file_ext = ".txt.gz"
        logger.info("Using pgzip with %d threads per file", args.pgzip_threads)
    except ImportError:
        logger.warning("pgzip not installed, falling back to standard gzip. Install with: pip install pgzip")
        open_func = lambda path: gzip.open(path, "wt")
        file_ext = ".txt.gz"
elif args.compress == "gzip":
    open_func = lambda path: gzip.open(path, "wt")
    file_ext = ".txt.gz"
else:  # txt
    open_func = lambda path: open(path, "w")
    file_ext = ".txt"

# Open input files
mspfile = open(args.msp)

# Handle both VCF and BCF files
if args.vcf.endswith('.bcf'):
    import subprocess
    logger.info("BCF file detected, using bcftools view")
    vcf_proc = subprocess.Popen(
        ['bcftools', 'view', '-Ov', args.vcf],
        stdout=subprocess.PIPE,
        text=True
    )
    vcf = vcf_proc.stdout
else:
    vcf = gzip.open(args.vcf, "rt")

# Create queues and threads for each output file
write_queues = {}
file_threads = {}

def file_writer_thread(file_path, write_queue):
    with open_func(file_path) as f:
        while True:
            try:
                data = write_queue.get(timeout=30)
                if data is None:  # Poison pill to stop thread
                    write_queue.task_done()
                    break
                f.write(data)
                write_queue.task_done()
            except queue.Empty:
                logger.warning("Writer thread for %s timed out waiting for data", file_path)
                continue

# Create output files and start writer threads
for i in range(args.num_ancs):
    dos_queue = queue.Queue(maxsize=args.queue_size)
    anc_queue = queue.Queue(maxsize=args.queue_size)

    write_queues[f"dos{i}"] = dos_queue
    write_queues[f"ancdos{i}"] = anc_queue

    dos_file = f"{args.out_pre}.anc{i}.dosage{file_ext}"
    anc_file = f"{args.out_pre}.anc{i}.hapcount{file_ext}"

    dos_thread = threading.Thread(
        target=file_writer_thread,
        args=(dos_file, dos_queue),
        name=f"dos_writer_{i}"
    )
    anc_thread = threading.Thread(
        target=file_writer_thread,
        args=(anc_file, anc_queue),
        name=f"anc_writer_{i}"
    )

    dos_thread.start()
    anc_thread.start()

    file_threads[f"dos{i}"] = dos_thread
    file_threads[f"ancdos{i}"] = anc_thread

logger.info("Started %d writer threads", len(file_threads))

window = ("", 0, 0)  # initialize the current window to check

logger.info("Processing VCF file: %s", args.vcf)
line_number = 0
for line in vcf:
    line_number += 1

    if line_number % 1000 == 0:
        logger.info("Processed %d lines", line_number)

    if line.startswith("##"):
        continue
    elif line.startswith("#"):
        # Write header into each output file via queues
        # line already ends with \n from the VCF file iterator
        anc_header = "\t".join([line.strip("# \”).split("\t", 9)[item] for item in [0, 1, 2, 3, 4, 9]])
        for i in range(args.num_ancs):
            write_queues[f"dos{i}"].put(anc_header)
            write_queues[f"ancdos{i}"].put(anc_header)
    else:
        # Entry format is ['chrom', 'pos', 'id', 'ref', 'alt', 'qual', 'filter', 'info', 'format', 'genotypes']
        row = line.split("\t", 9)

        dos_anc_out = "\t".join(row[:5])
        output_lines = {}
        for i in range(args.num_ancs):
            output_lines[f"dos{i}"] = dos_anc_out
            output_lines[f"ancdos{i}"] = dos_anc_out

        pos = int(row[1]); skip_line = False
        while not (row[0] == window[0] and (window[1] <= pos < window[2])):
            if row[0] == window[0] and window[1] > pos:
                skip_line = True
                break
            ancs = mspfile.readline()
            if ancs.startswith("#"):
                continue
            if not ancs:
                break
            # chm, spos, epos, sgpos, egpos, nsnps, calls
            ancs_entry = ancs.split("\t", 6)
            # Convert ancs_entry[6] (ancestries) directly to np.int8
            call_array = np.fromiter(ancs_entry[6], dtype='S1', count=len(ancs_entry[6]))[::2].astype(np.int8).reshape(-1, 2)

            window = (ancs_entry[0], int(ancs_entry[1]), int(ancs_entry[2]))
            if row[0] == window[0] and window[1] > pos:
                skip_line = True
                break

        if skip_line:
            logger.info("VCF position %d is not in an msp window, skipping site", pos)
            continue

        # Convert row[9] (genotypes) directly to np.int8
        geno_array = np.fromiter(row[9], dtype='S1', count=len(row[9]))[::2].astype(np.int8).reshape(-1, 2)

        # Create anc_mask: shape (n_individuals, 2_haplotypes, n_ancestries)
        anc_range = np.arange(args.num_ancs, dtype=np.int8)
        anc_mask = np.int8(call_array[:, :, None] == anc_range[None, None, :])

        # Count dosages and ancestries using the combined anc_mask
        dos_counts = np.sum(anc_mask * geno_array[:, :, None], axis=1)
        anc_counts = np.sum(anc_mask, axis=1)

        # Convert to string for writing
        dos_strs = np.apply_along_axis(lambda col: '\t'.join(col), 0, dos_counts.astype('U1'))
        anc_strs = np.apply_along_axis(lambda col: '\t'.join(col), 0, anc_counts.astype('U1'))

        for j in range(args.num_ancs):
            write_queues[f"dos{j}"].put(output_lines[f"dos{j}"] + "\t" + dos_strs[j] + "\n")
            write_queues[f"ancdos{j}"].put(output_lines[f"ancdos{j}"] + "\t" + anc_strs[j] + "\n")


# Signal all threads to stop by sending poison pills
for write_queue in write_queues.values():
    write_queue.put(None)

# Wait for all queues to be fully processed, then wait for threads to exit
for write_queue in write_queues.values():
    write_queue.join()

for file_thread in file_threads.values():
    file_thread.join()

# Close input file handles
mspfile.close()
vcf.close()

logger.info("Finished extracting tracts per %d ancestries!", args.num_ancs)