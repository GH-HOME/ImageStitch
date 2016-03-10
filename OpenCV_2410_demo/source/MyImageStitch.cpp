
#include "MyImageStitch.h"

cv::Mat ImageStitch(vector<cv::Mat>image_warpeds, vector<cv::Mat>mask_warpeds, vector<cv::Mat>mask_graphcut, vector<cv::Point>corners)
{
	// Default parameter
	bool try_gpu = false;
	int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
	int blend_type = Blender::MULTI_BAND;
	float blend_strength = 5;


	int num_images = image_warpeds.size();
	vector<Size> sizes(num_images);
	sizes[0] = image_warpeds[0].size();
	sizes[1] = image_warpeds[1].size();
	cout << sizes[0] << endl;



	Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
	compensator->feed(corners, image_warpeds, mask_warpeds);


	Mat img_warped, img_warped_s;
	Mat dilated_mask, seam_mask, mask, mask_warped;
	Ptr<Blender> blender;


	char str[1024];
	for (int img_idx = 0; img_idx < num_images; ++img_idx)
	{
		img_warped = image_warpeds[img_idx].clone();
		mask_warped = mask_warpeds[img_idx];
		// Compensate exposure
		//compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);


		img_warped.convertTo(img_warped_s, CV_16S);
		img_warped.release();

		dilate(mask_graphcut[img_idx], dilated_mask, Mat(15, 15, CV_8U));
		/*sprintf(str, "dilate_%d.png", img_idx);
		imwrite(str, dilated_mask);*/

		//resize(dilated_mask, seam_mask, mask_warped.size());
		mask_warped = dilated_mask & mask_warped;////////经过膨胀处理后的mask


		//mask_warped = mask_graphcut[img_idx].clone();

		if (blender.empty())
		{
			blender = Blender::createDefault(blend_type, try_gpu);
			Size dst_sz = resultRoi(corners, sizes).size();

			Rect test = resultRoi(corners, sizes);
			/*cout << "resultRoi    " << test << endl;
			cout << " dst_sz    " << dst_sz << endl;
			cout << "dst_sz.area()      " << dst_sz.area() << endl;*/
			float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
			cout << "blend_width    " << blend_width << endl;
			if (blend_width < 1.f)
				blender = Blender::createDefault(Blender::NO, try_gpu);
			else if (blend_type == Blender::MULTI_BAND)
			{
				MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(static_cast<Blender*>(blender));
				mb->setNumBands(static_cast<int>(ceil(log(blend_width) / log(2.)) - 1.));
				LOGLN("Multi-band blender, number of bands: " << mb->numBands());
			}
			else if (blend_type == Blender::FEATHER)
			{
				FeatherBlender* fb = dynamic_cast<FeatherBlender*>(static_cast<Blender*>(blender));
				fb->setSharpness(1.f / blend_width);
				LOGLN("Feather blender, sharpness: " << fb->sharpness());
			}
			blender->prepare(corners, sizes);
			/*cout << "blender->prepare sizes    " << sizes[0] << endl;
			cout << "corners      " << corners[0] << endl;
			cout << "blender->prepare sizes    " << sizes[1] << endl;
			cout << "corners      " << corners[1] << endl;*/

		}

		// Blend the current image
		blender->feed(img_warped_s, mask_warped, corners[img_idx]);
	}

	Mat result, result_mask;
	blender->blend(result, result_mask);

	//imwrite("result_mask.png", result_mask);
	LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

	result.convertTo(result, CV_8UC3);
	//imshow("result", result);
	//waitKey();
	return result;

	
}
