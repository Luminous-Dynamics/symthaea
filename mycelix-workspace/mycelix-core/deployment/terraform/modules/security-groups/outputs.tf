# Security Groups Module Outputs

output "rds_sg_id" {
  description = "ID of the RDS security group"
  value       = aws_security_group.rds.id
}

output "redis_sg_id" {
  description = "ID of the Redis security group"
  value       = aws_security_group.redis.id
}

output "validator_sg_id" {
  description = "ID of the validator security group"
  value       = aws_security_group.validator.id
}

output "fl_node_sg_id" {
  description = "ID of the FL node security group"
  value       = aws_security_group.fl_node.id
}

output "alb_sg_id" {
  description = "ID of the ALB security group"
  value       = aws_security_group.alb.id
}

output "bastion_sg_id" {
  description = "ID of the bastion security group"
  value       = aws_security_group.bastion.id
}
