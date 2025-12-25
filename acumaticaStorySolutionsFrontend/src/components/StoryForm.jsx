import React, { useState, useMemo } from 'react';
import {
  Box,
  TextField,
  Button,
  Paper,
  Typography,
  IconButton,
  Chip,
  Alert,
  Collapse,
  List,
  ListItem,
  ListItemText,
  CircularProgress,
  useTheme,
  alpha,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
  Stop as StopIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  ContentCopy as ContentCopyIcon,
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

/**
 * Parse acceptance criteria from various formats
 * Supports: bullets (-, *, •), numbers (1., 2., etc.), or plain newlines
 */
const parseAcceptanceCriteria = (text) => {
  if (!text || !text.trim()) {
    return [];
  }

  // Split by newlines
  const lines = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);

  const criteria = [];
  
  for (const line of lines) {
    // Remove common prefixes: bullets (-, *, •), numbers (1., 2., etc.), or dashes
    let cleaned = line
      .replace(/^[-*•]\s+/, '') // Remove bullet points
      .replace(/^\d+[.)]\s+/, '') // Remove numbered prefixes (1., 2., etc.)
      .replace(/^-\s+/, '') // Remove dash prefixes
      .trim();

    // Only add if there's actual content
    if (cleaned.length > 0) {
      criteria.push(cleaned);
    }
  }

  return criteria;
};

const StoryForm = ({ onSubmit, onCancel, isLoading, canCancel }) => {
  const theme = useTheme();
  const [formData, setFormData] = useState({
    story_id: '',
    title: '',
    description: '',
    acceptance_criteria_raw: '', // Single textarea input
  });
  const [errors, setErrors] = useState({});
  const [showPreview, setShowPreview] = useState(false);

  // Parse acceptance criteria from raw text
  const parsedCriteria = useMemo(() => {
    return parseAcceptanceCriteria(formData.acceptance_criteria_raw);
  }, [formData.acceptance_criteria_raw]);

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    }
    
    if (parsedCriteria.length === 0) {
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
    
    onSubmit({
      story_id: formData.story_id.trim() || null,
      title: formData.title.trim() || null,
      description: formData.description.trim(),
      acceptance_criteria: parsedCriteria,
    });
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
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        borderRadius: 2,
        transition: 'all 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
        overflow: 'hidden',
      }}
    >
      <Box
        sx={{
          p: { xs: 2, sm: 2.5 },
          pb: { xs: 1.5, sm: 2 },
          borderBottom: `1px solid ${theme.palette.divider}`,
          flexShrink: 0,
        }}
      >
        <Typography 
          variant="h5" 
          sx={{ 
            fontWeight: 600,
            fontSize: { xs: '1.25rem', sm: '1.5rem' },
          }}
        >
          Add Story Details
        </Typography>
      </Box>

      <Box 
        component="form" 
        onSubmit={handleSubmit} 
        sx={{ 
          flex: 1,
          display: 'flex', 
          flexDirection: 'column',
          overflow: 'hidden',
          minHeight: 0,
        }}
      >
        {/* Scrollable Content Area */}
        <Box
          sx={{
            flex: 1,
            overflowY: 'auto',
            overflowX: 'hidden',
            p: { xs: 2, sm: 2.5 },
            gap: { xs: 2, sm: 2.5 },
            display: 'flex',
            flexDirection: 'column',
            minHeight: 0,
            '&::-webkit-scrollbar': {
              width: '8px',
            },
            '&::-webkit-scrollbar-track': {
              background: 'transparent',
            },
            '&::-webkit-scrollbar-thumb': {
              background: theme.palette.mode === 'dark' 
                ? 'rgba(255, 255, 255, 0.2)' 
                : 'rgba(0, 0, 0, 0.2)',
              borderRadius: '4px',
              '&:hover': {
                background: theme.palette.mode === 'dark' 
                  ? 'rgba(255, 255, 255, 0.3)' 
                  : 'rgba(0, 0, 0, 0.3)',
              },
            },
          }}
        >
        {/* Story ID */}
        <TextField
          label="Story ID (Optional)"
          placeholder="e.g., STORY-001"
          value={formData.story_id}
          onChange={(e) => handleFieldChange('story_id', e.target.value)}
          fullWidth
          disabled={isLoading}
          size="small"
        />

        {/* Title */}
        <TextField
          label="Title (Optional)"
          placeholder="e.g., Sales Returns Processing"
          value={formData.title}
          onChange={(e) => handleFieldChange('title', e.target.value)}
          fullWidth
          disabled={isLoading}
          size="small"
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
          maxRows={10}
          required
          error={!!errors.description}
          helperText={errors.description || 'Enter the detailed description of the JIRA story'}
          disabled={isLoading}
          size="small"
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
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
              Acceptance Criteria *
            </Typography>
            {parsedCriteria.length > 0 && (
              <Chip
                label={`${parsedCriteria.length} criteria detected`}
                color="primary"
                size="small"
                variant="outlined"
              />
            )}
          </Box>

          <Alert severity="info" sx={{ mb: 1.5, py: { xs: 0.75, sm: 1 } }}>
            <Typography variant="body2" component="div" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
              <strong>💡 Tip:</strong> Copy acceptance criteria directly from JIRA and paste here. Supports:
              <Box component="ul" sx={{ mt: 0.5, mb: 0, pl: 2 }}>
                <li>Bullet points (-, *, •)</li>
                <li>Numbered lists (1., 2., etc.)</li>
                <li>Plain text (one per line)</li>
              </Box>
            </Typography>
          </Alert>

          {errors.acceptance_criteria && (
            <Alert severity="error" sx={{ mb: 1.5, py: { xs: 0.75, sm: 1 } }}>
              {errors.acceptance_criteria}
            </Alert>
          )}

          <TextField
            label="Acceptance Criteria"
            placeholder="Paste acceptance criteria here...&#10;&#10;Example formats:&#10;- Criterion 1&#10;- Criterion 2&#10;&#10;Or:&#10;1. Criterion 1&#10;2. Criterion 2&#10;&#10;Or just newlines:&#10;Criterion 1&#10;Criterion 2"
            value={formData.acceptance_criteria_raw}
            onChange={(e) => {
              handleFieldChange('acceptance_criteria_raw', e.target.value);
              // Auto-show preview when criteria are detected
              const parsed = parseAcceptanceCriteria(e.target.value);
              if (parsed.length > 0) {
                setShowPreview(true);
              }
            }}
            fullWidth
            multiline
            minRows={3}
            maxRows={8}
            required
            error={!!errors.acceptance_criteria}
            helperText={errors.acceptance_criteria || `Enter acceptance criteria (${parsedCriteria.length} detected)`}
            disabled={isLoading}
            size="small"
            sx={{
              '& .MuiOutlinedInput-root': {
                transition: 'all 0.2s ease-in-out',
                '& textarea': {
                  resize: 'vertical',
                  lineHeight: 1.6,
                  fontFamily: 'monospace',
                  fontSize: '0.9rem',
                },
              },
              '&:hover .MuiOutlinedInput-root': {
                borderColor: 'primary.main',
              },
            }}
          />

          {/* Preview of Parsed Criteria */}
          {parsedCriteria.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Button
                startIcon={showPreview ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                onClick={() => setShowPreview(!showPreview)}
                size="small"
                sx={{ mb: 1 }}
              >
                {showPreview ? 'Hide' : 'Show'} Preview ({parsedCriteria.length} criteria)
              </Button>
              
              <Collapse in={showPreview}>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 2,
                    bgcolor: 'background.default',
                    maxHeight: 300,
                    overflow: 'auto',
                  }}
                >
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Parsed criteria (will be sent to backend):
                  </Typography>
                  <List dense>
                    {parsedCriteria.map((criterion, index) => (
                      <ListItem
                        key={index}
                        sx={{
                          py: 0.5,
                          borderLeft: `3px solid`,
                          borderColor: 'primary.main',
                          pl: 2,
                          mb: 0.5,
                          bgcolor: 'background.paper',
                          borderRadius: 1,
                        }}
                      >
                        <Chip
                          label={index + 1}
                          size="small"
                          color="primary"
                          variant="outlined"
                          sx={{ mr: 1.5, minWidth: 32 }}
                        />
                        <ListItemText
                          primary={criterion}
                          primaryTypographyProps={{
                            variant: 'body2',
                            sx: { wordBreak: 'break-word' },
                          }}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Paper>
              </Collapse>
            </Box>
          )}
        </Box>

        </Box>

        {/* Action Buttons - Fixed at Bottom */}
        <Box 
          sx={{ 
            display: 'flex', 
            gap: { xs: 1.5, sm: 2 }, 
            bgcolor: 'background.paper',
            borderTop: `1px solid ${theme.palette.divider}`,
            p: { xs: 1.5, sm: 2 },
            flexShrink: 0,
          }}
        >
          <Button
            type="submit"
            variant="contained"
            size="medium"
            startIcon={isLoading ? <CircularProgress size={14} color="inherit" /> : <SendIcon />}
            disabled={isLoading}
            sx={{ 
              flex: 1, 
              py: { xs: 0.875, sm: 1 },
              px: { xs: 2, sm: 2.5 },
              borderRadius: 2,
              background: `linear-gradient(135deg, ${theme.palette.primary.main}, ${theme.palette.secondary?.main || theme.palette.primary.dark})`,
              boxShadow: `0 4px 12px ${alpha(theme.palette.primary.main, 0.4)}`,
              fontWeight: 600,
              fontSize: { xs: '0.875rem', sm: '0.9375rem' },
              textTransform: 'none',
              transition: 'all 0.3s ease',
              '&:hover': {
                transform: 'translateY(-2px)',
                boxShadow: `0 6px 16px ${alpha(theme.palette.primary.main, 0.5)}`,
              },
              '&:disabled': {
                background: theme.palette.action.disabledBackground,
              },
            }}
          >
            {isLoading ? 'Generating...' : 'Generate Solution'}
          </Button>
          
          {canCancel && (
            <Button
              variant="outlined"
              color="error"
              size="medium"
              startIcon={<StopIcon />}
              onClick={onCancel}
              sx={{ 
                py: { xs: 0.875, sm: 1 },
                px: { xs: 1.5, sm: 2 },
                borderRadius: 2,
                borderWidth: 2,
                fontWeight: 600,
                fontSize: { xs: '0.875rem', sm: '0.9375rem' },
                textTransform: 'none',
                '&:hover': {
                  borderWidth: 2,
                  transform: 'translateY(-2px)',
                },
              }}
            >
              Stop
            </Button>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

export default StoryForm;

