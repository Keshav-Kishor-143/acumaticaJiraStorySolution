/**
 * Story Normalization Utility
 * 
 * Normalizes raw JIRA story text into clean, canonical JSON structure
 * using the backend normalization endpoint.
 */

import axios from 'axios';
import API_CONFIG from '../api/config';
import toast from 'react-hot-toast';

/**
 * Normalize raw story text using backend LLM normalization
 * @param {string} rawStoryText - Raw story text from user
 * @returns {Promise<Object>} Normalized story structure
 */
export const normalizeStory = async (rawStoryText) => {
  try {
    const response = await axios.post(
      `${API_CONFIG.baseURL}${API_CONFIG.endpoints.solutions.normalize}`,
      {
        raw_story_text: rawStoryText
      },
      {
        headers: API_CONFIG.defaultHeaders,
        timeout: 60000 // 60 seconds for normalization (LLM can take time)
      }
    );

    console.log('Normalization API response:', response.data); // Debug log

    if (response.data && response.data.success) {
      const normalized = response.data.normalized_story;
      
      // Validate normalized structure
      if (!normalized || typeof normalized !== 'object') {
        console.error('Invalid normalized_story structure:', normalized);
        throw new Error('Invalid normalization response structure');
      }
      
      // Ensure required fields exist
      if (!normalized.hasOwnProperty('description') || 
          !normalized.hasOwnProperty('acceptance_criteria') ||
          !normalized.hasOwnProperty('requirements') ||
          !normalized.hasOwnProperty('story_title')) {
        console.error('Missing required fields in normalized_story:', normalized);
        throw new Error('Normalized story missing required fields');
      }
      
      console.log('Normalization successful:', {
        has_title: !!normalized.story_title,
        has_description: !!normalized.description,
        ac_count: normalized.acceptance_criteria?.length || 0,
        requirements_count: normalized.requirements?.length || 0
      });
      
      return normalized;
    } else {
      const errorMsg = response.data?.error || 'Normalization failed';
      console.error('Normalization API returned success=false:', errorMsg);
      throw new Error(errorMsg);
    }
  } catch (error) {
    // Check if it's a network/timeout error vs API error
    if (error.code === 'ECONNABORTED' || error.message.includes('timeout')) {
      console.error('Normalization timeout - request took too long:', error);
      toast.error('Normalization timed out. Please try again or use shorter text.');
      throw error; // Re-throw timeout errors so caller can handle
    }
    
    if (error.response) {
      // API returned an error response
      console.error('Normalization API error:', {
        status: error.response.status,
        data: error.response.data,
        error: error.message
      });
      
      // If it's a 422 or validation error, don't use fallback - show the error
      if (error.response.status === 422 || error.response.status === 400) {
        toast.error(`Normalization failed: ${error.response.data?.detail || error.response.data?.error || error.message}`);
        throw error;
      }
    } else if (error.request) {
      // Request was made but no response received
      console.error('Normalization network error - no response:', error);
      toast.error('Cannot connect to server. Please check if the backend is running.');
      throw error;
    } else {
      // Something else happened
      console.error('Story normalization error:', error);
    }
    
    // Only use fallback for unexpected errors, not for validation/network errors
    console.warn('Using fallback normalization due to error');
    return fallbackNormalization(rawStoryText);
  }
};

/**
 * Fallback normalization using simple parsing
 * Used when LLM normalization fails
 */
const fallbackNormalization = (rawStoryText) => {
  const lines = rawStoryText.split('\n');
  
  const descriptionParts = [];
  const acParts = [];
  let inAcSection = false;
  
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    
    // Detect AC section
    if (/acceptance\s+criteria|ac\s*:/i.test(trimmed)) {
      inAcSection = true;
      continue;
    }
    
    if (inAcSection) {
      // Remove bullets/numbers
      const cleaned = trimmed
        .replace(/^[-*•]\s+/, '')
        .replace(/^\d+[.)]\s+/, '')
        .trim();
      if (cleaned) {
        acParts.push(cleaned);
      }
    } else {
      descriptionParts.push(trimmed);
    }
  }
  
  return {
    story_title: '',
    description: descriptionParts.join('\n'),
    requirements: [],
    acceptance_criteria: acParts.map((ac, i) => ({
      id: `AC${i + 1}`,
      text: ac,
      subpoints: []
    }))
  };
};

/**
 * Convert normalized story to legacy format for backward compatibility
 * @param {Object} normalizedStory - Normalized story structure
 * @returns {Object} Legacy format { description, acceptance_criteria }
 */
export const normalizedToLegacyFormat = (normalizedStory) => {
  // Extract description
  const description = normalizedStory.description || '';
  
  // Extract acceptance criteria as flat list
  const acceptance_criteria = normalizedStory.acceptance_criteria?.map(ac => {
    // Combine main text with subpoints
    const mainText = ac.text || '';
    const subpoints = ac.subpoints || [];
    
    if (subpoints.length > 0) {
      return `${mainText}\n${subpoints.map(sp => `  - ${sp}`).join('\n')}`;
    }
    return mainText;
  }) || [];
  
  return {
    description,
    acceptance_criteria,
    story_title: normalizedStory.story_title || '',
    requirements: normalizedStory.requirements || []
  };
};

