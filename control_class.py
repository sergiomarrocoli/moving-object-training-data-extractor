from dependencies import *

class ControlClass:
    def compute():
        # set parameters
        percent_frames = 3
        k_size = 9
        # set files and paths
        videos_path = '/media/sergio/0C5EC1615EC14464/chimp_and_see/data/interim/videos/animals'
        export_path = '/media/sergio/0C5EC1615EC14464/msc_data/images/animals_demo'
        output_csv = "output_data_demo.csv"
        data_csv = "video_subject_species.csv"
        # get csv data
        filenames_and_subject_ids = pd.read_csv(data_csv)
        output_data = pd.read_csv(output_csv)

        # get counters and files to iterate over
        if len(output_data.image_id) > 1:
            image_id =  int(max(output_data.image_id)) + 1
            file_id =  int(max(output_data.file_id)) + 1
            frame_id =  int(max(output_data.frame_id)) + 1
        else:
            image_id = 1
            file_id = 1
            frame_id = 1

        print(image_id, file_id, frame_id)

        completed_files = output_data.filename.tolist()
        all_filenames = filenames_and_subject_ids.filename.tolist()
        remaining_files = list(set(all_filenames) - set(completed_files))
        existing_vids = os.listdir(videos_path)
        existing_vids = list(map(lambda x: x.split('.')[0], existing_vids)) # remove file extensions

        # make a dataframe
        data_store = []
        batch_counter = 0

        for filename in remaining_files:
            blank = True
            print(filename)
            species = filenames_and_subject_ids[filenames_and_subject_ids['filename'].isin([filename])].species.tolist()[0]
            print(species)
            if species != "human":
                subject_ids = filenames_and_subject_ids[filenames_and_subject_ids['filename'].isin([filename])].subject_id.tolist()

                # 1. get frames
                video_combiner = VideoCombiner(videos_path, subject_ids, percent_frames)
                raw_frames = video_combiner.get_frames()
        
                if len(raw_frames) > 0:
                    remove_outliers = RemoveOutliers(raw_frames, 0.25)
                    useable_frames = remove_outliers.get_frames()

                    # 2. process
                    image_processor = ImageProcessor(useable_frames, k_size)
                    boxes, _, _ = image_processor.analyse_frames()

                    len(sum(boxes, []))
                    if len(sum(boxes, [])) > 0:
                        blank = False
                        
                        # 3. export images
                        image_exporter = ImageExporter(useable_frames, boxes, export_path, image_id, frame_id)
                        image_id, frame_id, image_and_frame_ids = image_exporter.export_images()

                        # 4. write to df
                        for image in image_and_frame_ids:
                            # print([file_id, filename, species, image[0], image[1], image[2][0], image[2][1], image[2][2], image[2][3]])
                            data_store.append([file_id, filename, species, image[0], image[1], image[2][0], image[2][1], image[2][2], image[2][3]])
                            batch_counter += len(image_and_frame_ids)
        
            if blank == True:
                data_store.append([file_id, filename, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            file_id += 1

            # 5. write to csv
            if batch_counter > 500:
                with open(f"{output_csv}", 'a') as f:
                    writer = csv.writer(f)
                    for row in data_store:
                        writer.writerow(row)
                f.close
            # 6. clear data store/home/sergio/deep-learning/image_differencing/image_processing/dependencies.py
                data_store = []
                batch_counter = 0

ControlClass.compute()