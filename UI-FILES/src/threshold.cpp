#include <threshold.h>

void localThreshold(cv::Mat& src,cv::Mat& dst,int Wsize,TMode Tmode,Mode mode){

    for (int i = 0; i < src.rows - Wsize; i += Wsize){
        for (int j = 0; j < src.cols - Wsize; j += Wsize){
            cv::Mat ROI(cv::Size(Wsize,Wsize),CV_8UC1);
            for(int k = 0;k<Wsize;k++)
                for(int m = 0;m<Wsize;m++){
                    ROI.at<uchar>(k,m) = src.at<uchar>(i+k,j+m);
                }
            cv::Mat TROI = ROI;
            switch (Tmode) {
            case OTSU:
                otsu(ROI,TROI,mode);
                break;
            case OPTIMAL:
                optimalThreshold(ROI,TROI,mode);
                break;
            }

            for(int k = 0;k<Wsize;k++)
                for(int m = 0;m<Wsize;m++){
                    dst.at<uchar>(i+k,j+m) = TROI.at<uchar>(k,m);
                }

        }
    }
}

void optimalThreshold(cv::Mat& src,cv::Mat& dst,Mode mode){
    vector<int> background, foreground;
        int background_mean, foreground_mean, new_threshold, old_threshold = 0;
        for (int i = 0; i < src.rows; i++){
            for (int j = 0; j < src.cols; j++){
                if ((i == 0 && j == 0) || (i == 0 && j == src.cols - 1) || (i == src.rows - 1 && j == 0) || (i == src.rows - 1 && j == src.cols - 1))
                    background.push_back(src.at<uchar>(i, j));
                else
                    foreground.push_back(src.at<uchar>(i, j));
            }
        }
        background_mean = std::accumulate(background.begin(), background.end(), 0.0) / background.size();
        foreground_mean = std::accumulate(foreground.begin(), foreground.end(), 0.0) / foreground.size();
        new_threshold = (background_mean + foreground_mean) / 2;
        while (new_threshold != old_threshold)
        {
            old_threshold = new_threshold;
            for (int i = 0; i < src.rows; i++){
                for (int j = 0; j < src.cols; j++)
                    {
                        if (src.at<uchar>(i, j) < new_threshold)
                            background.push_back(src.at<uchar>(i, j));
                        else
                            foreground.push_back(src.at<uchar>(i, j));
                    }
                }
            background_mean = std::accumulate(background.begin(), background.end(), 0.0) / background.size();
            foreground_mean = std::accumulate(foreground.begin(), foreground.end(), 0.0) / foreground.size();
            new_threshold = (background_mean + foreground_mean) / 2;
        }
        int T = new_threshold;
        switch (mode){

        case BINARIZE:
            Binarize(src,dst,T);
            break;

        case MASK:
            Mask(src,dst,T);
            break;

        default:
            break;
        }
}

void otsu(cv::Mat& src,cv::Mat& dst,Mode mode){

    std::vector<int> hist(256,0);
    calcHist(src,hist);

    std::vector<float>variance;
    classVar(hist,variance);
    int T = calcThreshold(variance);

    switch (mode){

    case BINARIZE:
        Binarize(src,dst,T);
        break;

    case MASK:
        Mask(src,dst,T);
        break;

    default:
        break;
    }


}

void calcHist(cv::Mat& src,std::vector<int>& hist){
    for(int i=0;i<src.rows;i++){
        for(int j=0;j<src.cols;j++){
            int gray = src.at<uchar>(i,j);
            hist[gray]++;
        }
    }

}

void classVar(std::vector<int>& hist,std::vector<float>& variance){

    int sumFreq = std::accumulate(hist.begin(), hist.end(), 0);

    for(int seperator = 1; seperator<255; seperator++){
    float meanF =0 , wF = 0 ;
    float meanB =0 , wB = 0 ;

    for(int i =0;i<seperator;i++){
        meanF += i*hist[i];
        wF += hist[i];
    }
    meanF = meanF/wF;
    wF = wF/sumFreq;

    for(int i =seperator;i<255;i++){
        meanF += i*hist[i];
        wB += hist[i];
    }
    meanB = meanB/wB;
    wB = wF/sumFreq;

    float var = wB*wF*pow((meanB - meanF),2);
    variance.push_back(var);
    }
}

int calcThreshold(std::vector<float>& variance){
    int T =1;
    float max = variance[0];
    for(int i =1;i<variance.size();i++){
        if(variance[i] >= max){
            max = variance[i];
            T = i+1;
        }
    }
    return T;
}

void Binarize(cv::Mat& src,cv::Mat& dst,int T){
    for(int i =0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            if(src.at<uchar>(i,j)<T){
                dst.at<uchar>(i,j) =0;
            }
            else{
                dst.at<uchar>(i,j) =255;
            }
        }
    }
}

void Mask(cv::Mat& src,cv::Mat& dst,int T){
    for(int i =0;i<dst.rows;i++){
        for(int j=0;j<dst.cols;j++){
            if(src.at<uchar>(i,j)<T){
                dst.at<uchar>(i,j) =0;
            }
            else{
                dst.at<uchar>(i,j) =src.at<uchar>(i,j);
            }
        }
    }
}
