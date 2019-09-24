/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  if(is_initialized) return;

  num_particles = 100;  // TODO: Set the number of particles

  std::default_random_engine generator;

  // normal distributions for x, y, and theta
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y,std[1]);
  std::normal_distribution<double> dist_theta(theta,std[2]);

  // initialize all the particles
  for(int i = 0; i < num_particles; i++){
    Particle p;
    p.id = i;
    p.x = dist_x(generator);
    p.y = dist_y(generator);
    p.theta = dist_theta(generator);
    p.weight = 1.0;
    particles.push_back(p);
  }

  // set the flag for initialization, so only initialize once
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/ 
   */
   // create noise distribution
   std::normal_distribution<double> noise_x(0, std_pos[0]);
   std::normal_distribution<double> noise_y(0, std_pos[1]);
   std::normal_distribution<double> noise_theta(0, std_pos[2]);

   std::default_random_engine generator;

   for(auto& p : particles){
    if (fabs(yaw_rate) < 0.0001) {  
      p.x += velocity * delta_t * cos(p.theta);
      p.y += velocity * delta_t * sin(p.theta);
    } 
    else {
      p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate*delta_t) - sin(p.theta));
      p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate*delta_t));
      p.theta += yaw_rate * delta_t;
    }

    // add noise
    p.x += noise_x(generator);
    p.y += noise_y(generator);
    p.theta += noise_theta(generator);
   }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  for(auto& observation: observations){
    double min_distance = std::numeric_limits<float>::max();

    for(const auto& predict: predicted){
      double distance = dist(observation.x, observation.y, predict.x, predict.y);
      if( min_distance > distance){
        min_distance = distance;
        observation.id = predict.id;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for(auto& particle: particles){
     // reset weight
     particle.weight = 1.0;

    // create predictions for landmarks
    vector<LandmarkObs> predictions;
    for(const auto& landmark: map_landmarks.landmark_list){
      double distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
      if( distance < sensor_range){ 
        predictions.push_back(LandmarkObs{landmark.id_i, landmark.x_f, landmark.y_f});
      }
    }

    // transform
    vector<LandmarkObs> transformed_observations;
    double cos_theta = cos(particle.theta);
    double sin_theta = sin(particle.theta);

    for(const auto& obs: observations){
      LandmarkObs lmo;
      lmo.x = obs.x * cos_theta - obs.y * sin_theta + particle.x;
      lmo.y = obs.x * sin_theta + obs.y * cos_theta + particle.y;
      transformed_observations.push_back(lmo);
    }

    // associate
    dataAssociation(predictions, transformed_observations);

    // get weight
    for(const auto& transformed_observation: transformed_observations){

      Map::single_landmark_s landmark = map_landmarks.landmark_list.at(transformed_observation.id-1);
      double x_term = pow(transformed_observation.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
      double y_term = pow(transformed_observation.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
      double w = exp(-(x_term + y_term)) / (2 * M_PI * std_landmark[0] * std_landmark[1]);
      particle.weight *=  w;
    }

    weights.push_back(particle.weight);
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  // create uniform distribution for weights
    std::random_device rd;
    std::mt19937 generator(rd());
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    // create resampled particles
    vector<Particle> resampled_particles;
    for(auto& particle : particles){
      int idx = dist(generator);
      resampled_particles.push_back(particles[idx]);
    }

    // assign the resampled_particles to the previous particles
    particles = resampled_particles;

    // clear weight 
    weights.clear();
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}