import axios from 'axios';
import toast from 'react-hot-toast';
import API_CONFIG from './config';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_CONFIG.baseURL,
  timeout: API_CONFIG.timeout,
  headers: API_CONFIG.defaultHeaders
});

// Request interceptor for debugging
api.interceptors.request.use(
  (config) => {
    console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('❌ API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`✅ API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('❌ API Response Error:', error.response?.data || error.message);
    
    if (error.response?.status === 404) {
      toast.error('API endpoint not found. Please check if the backend is running.');
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Please try again later.');
    } else if (error.code === 'ECONNABORTED') {
      toast.error('Request timed out. The story processing is taking longer than expected. Please try again.');
    } else if (!error.response) {
      toast.error('Cannot connect to server. Please check if the backend is running on port 8001.');
    }
    
    return Promise.reject(error);
  }
);

/**
 * Process a JIRA story and generate a solution
 * @param {Object} storyData - Story data object
 * @param {string} storyData.description - JIRA story description (required)
 * @param {string[]} storyData.acceptance_criteria - List of acceptance criteria (required)
 * @param {string} [storyData.story_id] - Optional JIRA story ID
 * @param {string} [storyData.title] - Optional story title
 * @param {string[]} [storyData.images] - Optional list of image URLs or base64 images
 * @returns {Promise<Object>} Solution response
 */
export const processStory = async (storyData) => {
  try {
    // Create a separate axios instance for process endpoint with no timeout
    const processApi = axios.create({
      baseURL: API_CONFIG.baseURL,
      timeout: API_CONFIG.processTimeout || 0, // No timeout for story processing
      headers: API_CONFIG.defaultHeaders
    });
    
    // Add interceptors for process API
    processApi.interceptors.request.use(
      (config) => {
        console.log(`🚀 API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        console.error('❌ API Request Error:', error);
        return Promise.reject(error);
      }
    );
    
    processApi.interceptors.response.use(
      (response) => {
        console.log(`✅ API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        console.error('❌ API Response Error:', error.response?.data || error.message);
        
        if (error.response?.status === 404) {
          toast.error('API endpoint not found. Please check if the backend is running.');
        } else if (error.response?.status >= 500) {
          toast.error('Server error. Please try again later.');
        } else if (error.code === 'ECONNABORTED') {
          toast.error('Request timed out. The story processing is taking longer than expected. The backend may still be processing - please check backend logs.');
        } else if (!error.response) {
          toast.error('Cannot connect to server. Please check if the backend is running on port 8001.');
        }
        
        return Promise.reject(error);
      }
    );
    
    const response = await processApi.post(API_CONFIG.endpoints.solutions.process, {
      description: storyData.description,
      acceptance_criteria: storyData.acceptance_criteria,
      story_id: storyData.story_id || null,
      title: storyData.title || null,
      images: storyData.images || []
    });
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.response?.data?.error || error.message || 'Failed to process story';
    throw new Error(errorMessage);
  }
};

/**
 * Check health status of the solutions service
 * @returns {Promise<Object>} Health check response
 */
export const healthCheck = async () => {
  try {
    const response = await api.get(API_CONFIG.endpoints.solutions.health);
    return response.data;
  } catch (error) {
    throw new Error(`Health check failed: ${error.message}`);
  }
};

/**
 * Check general API health
 * @returns {Promise<Object>} Health check response
 */
export const apiHealthCheck = async () => {
  try {
    const response = await api.get(API_CONFIG.endpoints.health);
    return response.data;
  } catch (error) {
    throw new Error(`API health check failed: ${error.message}`);
  }
};

/**
 * Get list of available manuals from knowledge base
 * @returns {Promise<Object>} Manuals list response
 */
export const getManualsList = async () => {
  try {
    const response = await api.get(API_CONFIG.endpoints.solutions.manuals);
    return response.data;
  } catch (error) {
    const errorMessage = error.response?.data?.detail || error.response?.data?.error || error.message || 'Failed to fetch manuals list';
    throw new Error(errorMessage);
  }
};

// Utility functions for handling API responses
export const handleApiError = (error, defaultMessage = 'An error occurred') => {
  console.error('API Error:', error);
  
  if (error.response?.data?.detail) {
    toast.error(error.response.data.detail);
    return error.response.data.detail;
  } else if (error.response?.data?.error) {
    toast.error(error.response.data.error);
    return error.response.data.error;
  } else if (error.message) {
    toast.error(error.message);
    return error.message;
  } else {
    toast.error(defaultMessage);
    return defaultMessage;
  }
};

export const handleApiSuccess = (message) => {
  toast.success(message);
};

