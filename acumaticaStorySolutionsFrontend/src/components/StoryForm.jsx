import React, { useState, useMemo, useRef } from 'react';
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
  ToggleButton,
  ToggleButtonGroup,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Send as SendIcon,
  Stop as StopIcon,
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
  AutoAwesome as AutoAwesomeIcon,
} from '@mui/icons-material';
import { normalizeStory, normalizedToLegacyFormat } from '../utils/storyNormalizer';

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
  const [inputMode, setInputMode] = useState('structured'); // 'raw' or 'structured' - default to structured
  const [formData, setFormData] = useState({
    story_id: '',
    title: '',
    description: '',
    requirements_raw: '', // Requirements field for structured mode
    acceptance_criteria_raw: '', // Single textarea input (for backward compatibility)
    acceptance_criteria_items: [''], // Array of individual AC items for structured mode
    raw_story_text: '', // Raw story text for normalization
  });
  const [errors, setErrors] = useState({});
  const [showPreview, setShowPreview] = useState(false);
  const [isNormalizing, setIsNormalizing] = useState(false);
  const [normalizedPreview, setNormalizedPreview] = useState(null);
  const normalizationTimeoutRef = useRef(null);
  const previousValueRef = useRef('');

  // Parse acceptance criteria from raw text (for backward compatibility)
  const parsedCriteriaFromRaw = useMemo(() => {
    return parseAcceptanceCriteria(formData.acceptance_criteria_raw);
  }, [formData.acceptance_criteria_raw]);

  // Get parsed criteria - use individual items if available, otherwise parse from raw
  const parsedCriteria = useMemo(() => {
    // In structured mode, use individual AC items
    if (inputMode === 'structured') {
      return formData.acceptance_criteria_items
        .map(item => item.trim())
        .filter(item => item.length > 0);
    }
    // In raw mode, use parsed from raw text
    return parsedCriteriaFromRaw;
  }, [formData.acceptance_criteria_items, parsedCriteriaFromRaw, inputMode]);

  // Parse requirements from raw text (similar to AC parsing)
  const parsedRequirements = useMemo(() => {
    if (!formData.requirements_raw || !formData.requirements_raw.trim()) {
      return [];
    }
    return parseAcceptanceCriteria(formData.requirements_raw); // Reuse same parser
  }, [formData.requirements_raw]);

  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.description.trim()) {
      newErrors.description = 'Description is required';
    }
    
    // Validate AC items
    const validACItems = formData.acceptance_criteria_items.filter(item => item.trim().length > 0);
    if (validACItems.length === 0) {
      newErrors.acceptance_criteria = 'At least one acceptance criterion is required';
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Add new AC item field
  const handleAddACItem = () => {
    setFormData({
      ...formData,
      acceptance_criteria_items: [...formData.acceptance_criteria_items, '']
    });
  };

  // Remove AC item field
  const handleRemoveACItem = (index) => {
    if (formData.acceptance_criteria_items.length > 1) {
      const newItems = formData.acceptance_criteria_items.filter((_, i) => i !== index);
      setFormData({
        ...formData,
        acceptance_criteria_items: newItems
      });
    }
  };

  // Update individual AC item
  const handleACItemChange = (index, value) => {
    const newItems = [...formData.acceptance_criteria_items];
    newItems[index] = value;
    setFormData({
      ...formData,
      acceptance_criteria_items: newItems
    });
    
    // Clear error when user starts typing
    if (errors.acceptance_criteria) {
      setErrors({ ...errors, acceptance_criteria: null });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (inputMode === 'raw') {
      // Normalize raw story text
      if (!formData.raw_story_text.trim()) {
        setErrors({ raw_story_text: 'Raw story text is required' });
        return;
      }
      
      setIsNormalizing(true);
      try {
        console.log('Submitting story - normalizing text length:', formData.raw_story_text.length);
        const normalized = await normalizeStory(formData.raw_story_text);
        console.log('Normalization for submission completed:', normalized);
        
        // Validate normalized structure
        if (!normalized || typeof normalized !== 'object') {
          throw new Error('Invalid normalization response');
        }
        
        const legacyFormat = normalizedToLegacyFormat(normalized);
        console.log('Legacy format:', legacyFormat);
        
        // Use normalized data (extract title/story_id from normalized structure)
        onSubmit({
          story_id: normalized.story_title || null,  // Use extracted title as story_id if available
          title: normalized.story_title || null,
          description: legacyFormat.description,
          acceptance_criteria: legacyFormat.acceptance_criteria,
          requirements: legacyFormat.requirements || [],
          normalized_story: normalized, // Include normalized structure
        });
      } catch (error) {
        console.error('Normalization error on submit:', error);
        const errorMessage = error.message || 'Failed to normalize story. Please check your input.';
        setErrors({ raw_story_text: errorMessage });
        setIsNormalizing(false);
        return;
      } finally {
        setIsNormalizing(false);
      }
    } else {
      // Use structured input (existing flow)
      if (!validateForm()) {
        return;
      }
      
      // Use individual AC items from formData (backend will intelligently parse each)
      const acItems = formData.acceptance_criteria_items
        .map(item => item.trim())
        .filter(item => item.length > 0);
      
      // Build normalized structure from structured fields
      const normalizedStructure = {
        story_title: formData.title.trim() || '',
        description: formData.description.trim(),
        requirements: parsedRequirements,
        acceptance_criteria: acItems.map((ac, idx) => ({
          id: `AC${idx + 1}`,
          text: ac, // Backend will intelligently parse this (including Given/When/Then subpoints)
          subpoints: [] // Backend will extract subpoints if present in the text
        }))
      };
      
      onSubmit({
        story_id: formData.story_id.trim() || null,
        title: formData.title.trim() || null,
        description: formData.description.trim(),
        acceptance_criteria: acItems, // Send individual AC items (backend will parse intelligently)
        requirements: parsedRequirements, // Include requirements
        normalized_story: normalizedStructure, // Include normalized structure
      });
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
        {/* Input Mode Toggle */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, fontWeight: 600 }}>
            Input Mode
          </Typography>
          <ToggleButtonGroup
            value={inputMode}
            exclusive
            onChange={(e, newMode) => {
              if (newMode !== null) {
                setInputMode(newMode);
                setErrors({});
                setNormalizedPreview(null);
                // Initialize AC items array when switching to structured mode
                if (newMode === 'structured' && (!formData.acceptance_criteria_items || formData.acceptance_criteria_items.length === 0)) {
                  setFormData(prev => ({
                    ...prev,
                    acceptance_criteria_items: ['']
                  }));
                }
              }
            }}
            size="small"
            fullWidth
            sx={{
              '& .MuiToggleButton-root': {
                textTransform: 'none',
                fontWeight: 500,
              },
            }}
          >
            <ToggleButton value="structured">
              Structured Fields
            </ToggleButton>
            <ToggleButton value="raw">
              <AutoAwesomeIcon sx={{ mr: 1, fontSize: 18 }} />
              Raw Text (Auto-Normalize)
            </ToggleButton>
          </ToggleButtonGroup>
          {inputMode === 'raw' && (
            <Alert severity="info" sx={{ mt: 1.5 }}>
              <Typography variant="body2">
                <strong>💡 Smart Mode:</strong> Paste your complete JIRA story text here. 
                The system will automatically extract Description, Requirements, and Acceptance Criteria.
              </Typography>
            </Alert>
          )}
        </Box>

        {/* Raw Text Input Mode */}
        {inputMode === 'raw' && (
          <>
            {/* Loading indicator during normalization */}
            {isNormalizing && (
              <Alert severity="info" sx={{ mb: 2 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2">
                    <strong>Preparing normalized view...</strong> AI is analyzing your story text to extract Description, Requirements, and Acceptance Criteria.
                  </Typography>
                </Box>
              </Alert>
            )}

            <TextField
              label="Raw Story Text *"
              placeholder="Paste your complete JIRA story here...&#10;&#10;Example:&#10;As a sales user,&#10;I want the Email Quote template...&#10;&#10;Acceptance Criteria:&#10;1. The email subject must...&#10;2. The email body must include..."
              value={formData.raw_story_text}
              onChange={async (e) => {
                const value = e.target.value;
                const previousValue = previousValueRef.current;
                const textChange = value.length - previousValue.length;
                
                handleFieldChange('raw_story_text', value);
                previousValueRef.current = value;
                
                // Clear any pending normalization
                if (normalizationTimeoutRef.current) {
                  clearTimeout(normalizationTimeoutRef.current);
                  normalizationTimeoutRef.current = null;
                }
                
                // Clear preview when text is cleared
                if (value.trim().length === 0) {
                  setNormalizedPreview(null);
                  setIsNormalizing(false);
                  return;
                }
                
                // Auto-normalize on paste or typing - only if text is substantial
                if (value.length > 100) {
                  setIsNormalizing(true);
                  setNormalizedPreview(null); // Clear old preview while normalizing
                  
                  // Detect paste operation: large text change at once (>50 chars added)
                  const isPasteOperation = textChange > 50 || (previousValue.length < 50 && value.length > 200);
                  
                  // Use shorter delay for paste operations (immediate), longer for typing (debounced)
                  const debounceDelay = isPasteOperation ? 300 : 800;
                  
                  // Debounce normalization API call
                  normalizationTimeoutRef.current = setTimeout(async () => {
                    try {
                      console.log('Starting normalization for text length:', value.length);
                      const normalized = await normalizeStory(value);
                      console.log('Normalization completed:', normalized);
                      
                      // Validate normalized structure before using
                      if (!normalized || typeof normalized !== 'object') {
                        console.error('Invalid normalized structure received:', normalized);
                        setNormalizedPreview(null);
                        setIsNormalizing(false);
                        return;
                      }
                      
                      // Only update if the text hasn't changed during the API call
                      const currentValue = document.querySelector('[data-raw-story-input]')?.value;
                      if (currentValue === value) {
                        console.log('Setting normalized preview:', {
                          has_title: !!normalized.story_title,
                          has_description: !!normalized.description,
                          ac_count: normalized.acceptance_criteria?.length || 0
                        });
                        setNormalizedPreview(normalized);
                        setShowPreview(true); // Auto-expand preview when ready
                      } else {
                        console.log('Text changed during normalization, ignoring result');
                      }
                    } catch (err) {
                      console.error('Normalization error in form:', err);
                      // Only clear preview if it's a real error (not timeout/network)
                      // Timeout/network errors are already handled in normalizeStory
                      if (err.message && !err.message.includes('timeout') && !err.message.includes('network')) {
                        setNormalizedPreview(null);
                      }
                    } finally {
                      setIsNormalizing(false);
                      normalizationTimeoutRef.current = null;
                    }
                  }, debounceDelay);
                } else {
                  setIsNormalizing(false);
                  setNormalizedPreview(null);
                }
              }}
              inputProps={{ 'data-raw-story-input': true }}
              fullWidth
              multiline
              minRows={8}
              maxRows={15}
              required
              error={!!errors.raw_story_text}
              helperText={errors.raw_story_text || (isNormalizing ? 'AI is analyzing your story...' : 'Paste complete JIRA story text. AI will automatically extract Description, Requirements, and Acceptance Criteria.')}
              disabled={isLoading}
              size="small"
              sx={{
                '& .MuiOutlinedInput-root': {
                  '& textarea': {
                    fontFamily: 'monospace',
                    fontSize: '0.9rem',
                    lineHeight: 1.6,
                  },
                },
              }}
            />

            {/* Normalized Preview */}
            {normalizedPreview && (
              <Box sx={{ mt: 2 }}>
                <Button
                  startIcon={showPreview ? <ExpandLessIcon /> : <ExpandMoreIcon />}
                  onClick={() => setShowPreview(!showPreview)}
                  size="small"
                  sx={{ mb: 1 }}
                >
                  {showPreview ? 'Hide' : 'Show'} Normalized Preview
                </Button>
                
                <Collapse in={showPreview}>
                  <Paper variant="outlined" sx={{ p: 2, bgcolor: 'background.default' }}>
                    <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                      Normalized structure (will be sent to backend):
                    </Typography>
                    
                    {/* Description */}
                    <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                      Description:
                    </Typography>
                    <Typography variant="body2" sx={{ mb: 2, pl: 2, color: 'text.secondary', whiteSpace: 'pre-wrap' }}>
                      {normalizedPreview.description || '(empty)'}
                    </Typography>
                    
                    {/* Requirements */}
                    {normalizedPreview.requirements && normalizedPreview.requirements.length > 0 && (
                      <>
                        <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                          Requirements ({normalizedPreview.requirements.length}):
                        </Typography>
                        <List dense sx={{ mb: 2, pl: 2 }}>
                          {normalizedPreview.requirements.map((req, idx) => (
                            <ListItem key={idx} sx={{ py: 0.25, pl: 0 }}>
                              <Typography variant="body2" color="text.secondary">
                                • {req}
                              </Typography>
                            </ListItem>
                          ))}
                        </List>
                      </>
                    )}
                    
                    {/* Acceptance Criteria */}
                    <Typography variant="body2" sx={{ mb: 1, fontWeight: 600 }}>
                      Acceptance Criteria ({normalizedPreview.acceptance_criteria?.length || 0}):
                    </Typography>
                    <List dense>
                      {normalizedPreview.acceptance_criteria?.map((ac, idx) => (
                        <ListItem 
                          key={idx} 
                          sx={{ 
                            py: 0.5, 
                            pl: 2,
                            flexDirection: 'column',
                            alignItems: 'flex-start',
                            borderLeft: `3px solid ${theme.palette.primary.main}`,
                            mb: 1,
                            bgcolor: 'background.paper',
                            borderRadius: 1,
                          }}
                        >
                          <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: ac.subpoints?.length > 0 ? 0.5 : 0 }}>
                            <Chip label={ac.id} size="small" color="primary" variant="outlined" sx={{ mr: 1.5, minWidth: 40 }} />
                            <Typography variant="body2" sx={{ fontWeight: 600, flex: 1 }}>
                              {ac.text}
                            </Typography>
                          </Box>
                          {ac.subpoints && ac.subpoints.length > 0 && (
                            <Box sx={{ pl: 6, width: '100%' }}>
                              {ac.subpoints.map((subpoint, spIdx) => (
                                <Typography 
                                  key={spIdx} 
                                  variant="body2" 
                                  sx={{ 
                                    color: 'text.secondary',
                                    fontSize: '0.85rem',
                                    mb: 0.25,
                                    pl: 1,
                                    borderLeft: `2px solid ${theme.palette.divider}`
                                  }}
                                >
                                  • {subpoint}
                                </Typography>
                              ))}
                            </Box>
                          )}
                        </ListItem>
                      ))}
                    </List>
                  </Paper>
                </Collapse>
              </Box>
            )}
          </>
        )}

        {/* Structured Input Mode */}
        {inputMode === 'structured' && (
          <>
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
          placeholder="Enter the JIRA story description (As a... I want... So that...)..."
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

        {/* Requirements */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
              Requirements (Optional)
            </Typography>
            {parsedRequirements.length > 0 && (
              <Chip
                label={`${parsedRequirements.length} requirements detected`}
                color="secondary"
                size="small"
                variant="outlined"
              />
            )}
          </Box>

          <Alert severity="info" sx={{ mb: 1.5, py: { xs: 0.75, sm: 1 } }}>
            <Typography variant="body2" component="div" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
              <strong>💡 Tip:</strong> Enter business rules, constraints, or requirements. Supports:
              <Box component="ul" sx={{ mt: 0.5, mb: 0, pl: 2 }}>
                <li>Bullet points (-, *, •)</li>
                <li>Numbered lists (1., 2., etc.)</li>
                <li>Plain text (one per line)</li>
              </Box>
            </Typography>
          </Alert>

          <TextField
            label="Requirements"
            placeholder="Enter business rules and requirements...&#10;&#10;Example:&#10;- Event Start Date must be greater than or equal to Scheduled Start Date&#10;- Event End Date must be less than or equal to the Scheduled Return Date"
            value={formData.requirements_raw}
            onChange={(e) => {
              handleFieldChange('requirements_raw', e.target.value);
              // Auto-show preview when requirements are detected
              const parsed = parseAcceptanceCriteria(e.target.value); // Reuse same parser
              if (parsed.length > 0 && !showPreview) {
                setShowPreview(true);
              }
            }}
            fullWidth
            multiline
            minRows={2}
            maxRows={6}
            helperText={`Enter requirements (${parsedRequirements.length} detected)`}
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

          {/* Preview of Parsed Requirements */}
          {parsedRequirements.length > 0 && (
            <Box sx={{ mt: 1.5 }}>
              <Collapse in={showPreview}>
                <Paper
                  variant="outlined"
                  sx={{
                    p: 1.5,
                    bgcolor: 'background.default',
                    maxHeight: 200,
                    overflow: 'auto',
                  }}
                >
                  <Typography variant="caption" color="text.secondary" sx={{ mb: 1, display: 'block' }}>
                    Parsed requirements:
                  </Typography>
                  <List dense>
                    {parsedRequirements.map((req, index) => (
                      <ListItem
                        key={index}
                        sx={{
                          py: 0.25,
                          borderLeft: `3px solid`,
                          borderColor: 'secondary.main',
                          pl: 1.5,
                          mb: 0.25,
                          bgcolor: 'background.paper',
                          borderRadius: 1,
                        }}
                      >
                        <Chip
                          label={index + 1}
                          size="small"
                          color="secondary"
                          variant="outlined"
                          sx={{ mr: 1.5, minWidth: 32 }}
                        />
                        <ListItemText
                          primary={req}
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

        {/* Acceptance Criteria - Manual Entry (One at a Time) */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1.5 }}>
            <Typography variant="subtitle2" sx={{ fontWeight: 600, fontSize: { xs: '0.875rem', sm: '0.9375rem' } }}>
              Acceptance Criteria * ({parsedCriteria.length} {parsedCriteria.length === 1 ? 'criterion' : 'criteria'})
            </Typography>
            <Button
              startIcon={<AddIcon />}
              onClick={handleAddACItem}
              size="small"
              variant="outlined"
              color="primary"
              disabled={isLoading}
              sx={{ textTransform: 'none' }}
            >
              Add AC
            </Button>
          </Box>

          <Alert severity="info" sx={{ mb: 1.5, py: { xs: 0.75, sm: 1 } }}>
            <Typography variant="body2" component="div" sx={{ fontSize: { xs: '0.8rem', sm: '0.875rem' } }}>
              <strong>💡 Manual Entry:</strong> Add acceptance criteria one at a time. 
              Each field can contain a complete criterion (including subpoints if needed).
              The backend will intelligently process and structure them.
            </Typography>
          </Alert>

          {errors.acceptance_criteria && (
            <Alert severity="error" sx={{ mb: 1.5, py: { xs: 0.75, sm: 1 } }}>
              {errors.acceptance_criteria}
            </Alert>
          )}

          {/* Individual AC Item Fields */}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
            {formData.acceptance_criteria_items.map((acItem, index) => (
              <Box
                key={index}
                sx={{
                  display: 'flex',
                  gap: 1,
                  alignItems: 'flex-start',
                  p: 1.5,
                  border: `1px solid ${theme.palette.divider}`,
                  borderRadius: 1,
                  bgcolor: 'background.paper',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    borderColor: 'primary.main',
                    boxShadow: `0 2px 8px ${alpha(theme.palette.primary.main, 0.1)}`,
                  },
                }}
              >
                <Chip
                  label={`AC${index + 1}`}
                  size="small"
                  color="primary"
                  variant="outlined"
                  sx={{ 
                    minWidth: 50,
                    flexShrink: 0,
                    mt: 0.5
                  }}
                />
                <TextField
                  label={`Acceptance Criterion ${index + 1}`}
                  placeholder={`Enter acceptance criterion ${index + 1}...&#10;&#10;Example:&#10;Rule 1 – Event Start Date Validation&#10;Given a User is creating an Event Order&#10;When the User enters an Event Start Date&#10;Then the Event Start Date must be greater than or equal to the Scheduled Start Date`}
                  value={acItem}
                  onChange={(e) => handleACItemChange(index, e.target.value)}
                  fullWidth
                  multiline
                  minRows={2}
                  maxRows={6}
                  required={index === 0}
                  disabled={isLoading}
                  size="small"
                  sx={{
                    flex: 1,
                    '& .MuiOutlinedInput-root': {
                      transition: 'all 0.2s ease-in-out',
                      '& textarea': {
                        resize: 'vertical',
                        lineHeight: 1.6,
                      },
                    },
                    '&:hover .MuiOutlinedInput-root': {
                      borderColor: 'primary.main',
                    },
                  }}
                />
                {formData.acceptance_criteria_items.length > 1 && (
                  <IconButton
                    onClick={() => handleRemoveACItem(index)}
                    size="small"
                    color="error"
                    disabled={isLoading}
                    sx={{
                      flexShrink: 0,
                      mt: 0.5,
                      '&:hover': {
                        bgcolor: alpha(theme.palette.error.main, 0.1),
                      },
                    }}
                  >
                    <DeleteIcon fontSize="small" />
                  </IconButton>
                )}
              </Box>
            ))}
          </Box>

          {/* Helper text */}
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            💡 Tip: You can paste multi-line criteria (including Given/When/Then) into each field. 
            The backend will intelligently parse and structure them.
          </Typography>
        </Box>
          </>
        )}

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
            startIcon={(isLoading || isNormalizing) ? <CircularProgress size={14} color="inherit" /> : <SendIcon />}
            disabled={isLoading || isNormalizing}
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
            {(isLoading || isNormalizing) ? (isNormalizing ? 'Normalizing...' : 'Generating...') : 'Generate Solution'}
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
