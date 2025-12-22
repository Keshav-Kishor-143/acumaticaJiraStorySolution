import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  IconButton,
  Chip,
  Alert,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
} from '@mui/icons-material';

// Auto-resizing TextArea component with smart height adjustment
const AutoResizeTextArea = ({ value, onChange, placeholder, disabled, minRows = 1, maxRows = 8 }) => {
  return (
    <TextField
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      disabled={disabled}
      multiline
      minRows={minRows}
      maxRows={maxRows}
      fullWidth
      size="small"
      sx={{
        '& .MuiOutlinedInput-root': {
          transition: 'all 0.2s ease-in-out',
          '& textarea': {
            resize: 'none',
            overflow: 'hidden',
            lineHeight: 1.6,
            padding: '8.5px 14px',
          },
        },
        '&:hover .MuiOutlinedInput-root': {
          borderColor: 'primary.main',
        },
        '& .MuiOutlinedInput-root.Mui-focused': {
          borderColor: 'primary.main',
        },
      }}
    />
  );
};

const StoryForm = ({ onSubmit, isLoading }) => {
  const [formData, setFormData] = useState({
    story_id: '',
    title: '',
    description: '',
    acceptance_criteria: [''],
  });
  const [errors, setErrors] = useState({});

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    }
    
    const validCriteria = formData.acceptance_criteria.filter(c => c.trim());
    if (validCriteria.length === 0) {
      newErrors.acceptance_criteria = 'At least one acceptance criterion is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }
    
    const validCriteria = formData.acceptance_criteria.filter(c => c.trim());
    
    onSubmit({
      story_id: formData.story_id.trim() || null,
      title: formData.title.trim() || null,
      description: formData.description.trim(),
      acceptance_criteria: validCriteria,
    });
  };

  const handleAddCriterion = () => {
    setFormData({
      ...formData,
      acceptance_criteria: [...formData.acceptance_criteria, ''],
    });
  };

  const handleRemoveCriterion = (index) => {
    if (formData.acceptance_criteria.length > 1) {
      const newCriteria = formData.acceptance_criteria.filter((_, i) => i !== index);
      setFormData({
        ...formData,
        acceptance_criteria: newCriteria,
      });
    }
  };

  const handleCriterionChange = (index, value) => {
    const newCriteria = [...formData.acceptance_criteria];
    newCriteria[index] = value;
    setFormData({
      ...formData,
      acceptance_criteria: newCriteria,
    });
    
    // Clear error when user starts typing
    if (errors.acceptance_criteria) {
      setErrors({ ...errors, acceptance_criteria: null });
    }
  };

  const handleFieldChange = (field, value) => {
    setFormData({
      ...formData,
      [field]: value,
    });
    
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors({ ...errors, [field]: null });
    }
  };

  return (
    <Paper
      elevation={2}
      sx={{
        p: 3,
        borderRadius: 2,
      }}
    >
      <Typography variant="h5" gutterBottom sx={{ mb: 3, fontWeight: 600 }}>
        Process JIRA Story
      </Typography>

      <Box component="form" onSubmit={handleSubmit} sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
        {/* Story ID */}
        <TextField
          label="Story ID (Optional)"
          placeholder="e.g., STORY-001"
          value={formData.story_id}
          onChange={(e) => handleFieldChange('story_id', e.target.value)}
          fullWidth
          disabled={isLoading}
        />

        {/* Title */}
        <TextField
          label="Title (Optional)"
          placeholder="e.g., Sales Returns Processing"
          value={formData.title}
          onChange={(e) => handleFieldChange('title', e.target.value)}
          fullWidth
          disabled={isLoading}
        />

        {/* Description */}
        <TextField
          label="Description *"
          placeholder="Enter the JIRA story description..."
          value={formData.description}
          onChange={(e) => handleFieldChange('description', e.target.value)}
          fullWidth
          multiline
          minRows={3}
          maxRows={15}
          required
          error={!!errors.description}
          helperText={errors.description || 'Enter the detailed description of the JIRA story'}
          disabled={isLoading}
          sx={{
            '& .MuiOutlinedInput-root': {
              transition: 'all 0.2s ease-in-out',
              '& textarea': {
                resize: 'none',
                overflow: 'hidden',
                lineHeight: 1.6,
              },
            },
            '&:hover .MuiOutlinedInput-root': {
              borderColor: 'primary.main',
            },
          }}
        />

        {/* Acceptance Criteria */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
              Acceptance Criteria *
            </Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={handleAddCriterion}
              size="small"
              disabled={isLoading}
            >
              Add Criterion
            </Button>
          </Box>

          {errors.acceptance_criteria && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {errors.acceptance_criteria}
            </Alert>
          )}

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {formData.acceptance_criteria.map((criterion, index) => (
              <Box 
                key={index} 
                sx={{ 
                  display: 'flex', 
                  gap: 1.5, 
                  alignItems: 'flex-start',
                  transition: 'all 0.2s ease-in-out',
                  '&:hover': {
                    '& .criterion-chip': {
                      transform: 'scale(1.05)',
                    },
                  },
                }}
              >
                <Chip
                  className="criterion-chip"
                  label={index + 1}
                  size="small"
                  sx={{ 
                    minWidth: 40,
                    height: 'auto',
                    py: 0.5,
                    transition: 'transform 0.2s ease-in-out',
                    flexShrink: 0,
                    mt: 0.5,
                  }}
                  color="primary"
                  variant="outlined"
                />
                <Box sx={{ flex: 1, minWidth: 0 }}>
                  <AutoResizeTextArea
                    placeholder={`Enter acceptance criterion ${index + 1}...`}
                    value={criterion}
                    onChange={(e) => handleCriterionChange(index, e.target.value)}
                    disabled={isLoading}
                    minRows={1}
                    maxRows={8}
                  />
                </Box>
                {formData.acceptance_criteria.length > 1 && (
                  <IconButton
                    onClick={() => handleRemoveCriterion(index)}
                    disabled={isLoading}
                    color="error"
                    size="small"
                    sx={{
                      flexShrink: 0,
                      mt: 0.5,
                      transition: 'all 0.2s ease-in-out',
                      '&:hover': {
                        transform: 'scale(1.1)',
                        bgcolor: 'error.light',
                      },
                    }}
                  >
                    <DeleteIcon />
                  </IconButton>
                )}
              </Box>
            ))}
          </Box>
        </Box>

        {/* Submit Button */}
        <Button
          type="submit"
          variant="contained"
          color="primary"
          size="large"
          startIcon={<SendIcon />}
          disabled={isLoading}
          sx={{ mt: 2, py: 1.5 }}
        >
          {isLoading ? 'Processing...' : 'Generate Solution'}
        </Button>
      </Box>
    </Paper>
  );
};

export default StoryForm;

